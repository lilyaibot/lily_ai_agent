package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

// --- CONFIGURATION ---
const (
	DEFAULT_MAIN_LLM_URL   = "http://localhost:11434"
	DEFAULT_MAIN_LLM_MODEL = "llama3"
	WEB_PORT               = "8080"
	SOURCE_FILE            = "agent.go"
	MAX_LOOPS              = 30
	AUDIT_LOG              = "audit.log"
	MEMORY_FILE            = "memory.json"
	CONFIG_FILE            = "config.json"
	SESSIONS_FILE          = "chat_sessions.json"
	TASKS_FILE             = "tasks.json"
	DOCS_DIR               = "documents"
	NN_DIM                 = 256
	NN_FILE_PREFIX         = "nn_"
	NN_DEFAULT             = "agent_memory"     // primary episodic/general store
	NN_IDENTITY            = "agent_identity"   // personality, learned self-knowledge
	NN_SEMANTIC            = "agent_semantic"   // facts, user prefs, world knowledge
	NN_PROCEDURAL          = "agent_procedural" // workflows, how-to patterns
	NN_CODE                = "agent_code"       // coding patterns, solutions
	HEARTBEAT_INTERVAL     = 30 * time.Minute
)

// --- DATA STRUCTURES ---
type NNEntry struct {
	Key      string    `json:"key"`
	Value    string    `json:"value"`
	StoredAt time.Time `json:"stored_at"`
	Recalls  int       `json:"recalls"`
}

type NeuralNetwork struct {
	Name         string      `json:"name"`
	Capacity     int         `json:"capacity"`
	Stored       int         `json:"stored"`
	W            [][]float64 `json:"w"`
	Keys         []NNEntry   `json:"keys"`
	CreatedAt    time.Time   `json:"created_at"`
	UpdatedAt    time.Time   `json:"updated_at"`
	TotalRecalls int         `json:"total_recalls"`
}

type NNController struct {
	mu   sync.RWMutex
	nets map[string]*NeuralNetwork
}

var nnCtrl = &NNController{nets: make(map[string]*NeuralNetwork)}
var nnDir string

// Brain struct for memory operations
type Brain struct{}

var brain Brain

// SelfAgent represents a spawned agent instance
type SelfAgent struct {
	ID        string    `json:"id"`
	Role      string    `json:"role"`
	Goal      string    `json:"goal"`
	Status    string    `json:"status"`
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
}

var selfAgents []*SelfAgent

// Brain methods
func (b Brain) Remember(description, result string) {
	nnCtrl.store(NN_PROCEDURAL, description, result)
}

func (b Brain) StoreProcedural(name, steps string) {
	nnCtrl.store(NN_PROCEDURAL, name, steps)
}

// SelfAgent management
func spawnSelfAgent(role, goal, source string) *SelfAgent {
	mu.Lock()
	defer mu.Unlock()
	agent := &SelfAgent{
		ID:        fmt.Sprintf("agent_%d", time.Now().UnixNano()),
		Role:      role,
		Goal:      goal,
		Status:    "running",
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}
	selfAgents = append(selfAgents, agent)
	return agent
}

func listSelfAgents() []*SelfAgent {
	mu.Lock()
	defer mu.Unlock()
	return selfAgents
}

func stopSelfAgent(id string) {
	mu.Lock()
	defer mu.Unlock()
	for _, agent := range selfAgents {
		if agent.ID == id {
			agent.Status = "stopped"
			agent.UpdatedAt = time.Now()
			break
		}
	}
}

func nnInitDir() {
	exe, err := os.Executable()
	if err != nil {
		nnDir = "."
		return
	}
	resolved, err := filepath.EvalSymlinks(exe)
	if err != nil {
		resolved = exe
	}
	nnDir = filepath.Dir(resolved)
	auditLog("NEURAL", fmt.Sprintf("NN storage directory: %s", nnDir))
}

// ── Vector math ──────────────────────────────────────────────────────────────
func textToVec(text string) []float64 {
	v := make([]float64, NN_DIM)
	text = strings.ToLower(strings.TrimSpace(text))
	if text == "" {
		return v
	}
	padded := "  " + text + "  "
	runes := []rune(padded)
	for i := 0; i+2 < len(runes); i++ {
		h := uint32(2166136261)
		h ^= uint32(runes[i])
		h *= 16777619
		h ^= uint32(runes[i+1])
		h *= 16777619
		h ^= uint32(runes[i+2])
		h *= 16777619
		v[int(h%uint32(NN_DIM))] += 1.0
	}
	for _, w := range strings.Fields(text) {
		h := uint32(2166136261)
		for _, c := range w {
			h ^= uint32(c)
			h *= 16777619
		}
		v[int(h%uint32(NN_DIM))] += 2.0
	}
	return nnNormalize(v)
}

func nnNormalize(v []float64) []float64 {
	var sum float64
	for _, x := range v {
		sum += x * x
	}
	if sum == 0 {
		return v
	}
	mag := 1.0 / math.Sqrt(sum)
	out := make([]float64, len(v))
	for i, x := range v {
		out[i] = x * mag
	}
	return out
}

func nnDot(a, b []float64) float64 {
	var s float64
	for i := range a {
		if i < len(b) {
			s += a[i] * b[i]
		}
	}
	return s
}

func nnOuterAdd(W [][]float64, a, b []float64, scale float64) {
	for i := range a {
		for j := range b {
			W[i][j] += scale * a[i] * b[j]
		}
	}
}

func newWeightMatrix() [][]float64 {
	W := make([][]float64, NN_DIM)
	for i := range W {
		W[i] = make([]float64, NN_DIM)
	}
	return W
}

// ── Network lifecycle ─────────────────────────────────────────────────────────
func nnFilePath(name string) string {
	if nnDir == "" {
		nnDir = "."
	}
	return filepath.Join(nnDir, NN_FILE_PREFIX+name+".json")
}

func isValidNNName(name string) bool {
	for _, c := range name {
		if !((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_') {
			return false
		}
	}
	return len(name) > 0 && len(name) <= 40
}

func (c *NNController) spawn(name string, capacity int) *NeuralNetwork {
	c.mu.Lock()
	defer c.mu.Unlock()
	if nn, ok := c.nets[name]; ok {
		return nn
	}
	if capacity <= 0 || capacity > 50000 {
		capacity = 500
	}
	nn := &NeuralNetwork{
		Name:      name,
		Capacity:  capacity,
		W:         newWeightMatrix(),
		Keys:      []NNEntry{},
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}
	if data, err := os.ReadFile(nnFilePath(name)); err == nil {
		var loaded NeuralNetwork
		if json.Unmarshal(data, &loaded) == nil {
			nn = &loaded
		}
	}
	c.nets[name] = nn
	auditLog("NEURAL", fmt.Sprintf("Network '%s' online (cap:%d stored:%d)", name, nn.Capacity, nn.Stored))
	return nn
}

func (c *NNController) get(name string) (*NeuralNetwork, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	nn, ok := c.nets[name]
	return nn, ok
}

func (c *NNController) save(name string) {
	c.mu.RLock()
	nn, ok := c.nets[name]
	c.mu.RUnlock()
	if !ok {
		return
	}
	nn.UpdatedAt = time.Now()
	data, _ := json.MarshalIndent(nn, "", "  ")
	os.WriteFile(nnFilePath(name), data, 0644)
}

func (c *NNController) listAll() []map[string]interface{} {
	c.mu.RLock()
	defer c.mu.RUnlock()
	var result []map[string]interface{}
	for _, nn := range c.nets {
		util := 0.0
		if nn.Capacity > 0 {
			util = float64(nn.Stored) / float64(nn.Capacity) * 100
		}
		result = append(result, map[string]interface{}{
			"name":      nn.Name,
			"capacity":  nn.Capacity,
			"stored":    nn.Stored,
			"util_pct":  util,
			"recalls":   nn.TotalRecalls,
			"created":   nn.CreatedAt.Format("2006-01-02 15:04"),
			"updated":   nn.UpdatedAt.Format("2006-01-02 15:04"),
		})
	}
	return result
}

// ── Core ops ─────────────────────────────────────────────────────────────────
func (c *NNController) store(name, key, value string) string {
	if !isValidNNName(name) {
		return "Error: invalid network name."
	}
	nn, ok := c.get(name)
	if !ok {
		nn = c.spawn(name, 500)
	}
	util := float64(nn.Stored) / float64(nn.Capacity)
	scale := 1.0
	if util > 0.8 {
		decay := 1.0 - (util-0.8)*0.5
		for i := range nn.W {
			for j := range nn.W[i] {
				nn.W[i][j] *= decay
			}
		}
	}
	nnOuterAdd(nn.W, textToVec(key), textToVec(value), scale)
	nn.Keys = append(nn.Keys, NNEntry{Key: key, Value: value, StoredAt: time.Now()})
	nn.Stored++
	go c.save(name)
	auditLog("NEURAL", fmt.Sprintf("Stored in '%s': %.50s", name, key))
	return fmt.Sprintf("Stored in '%s'. Utilization: %.0f%%", name, float64(nn.Stored)/float64(nn.Capacity)*100)
}

func (c *NNController) recall(name, query string, topK int) string {
	nn, ok := c.get(name)
	if !ok {
		return fmt.Sprintf("Network '%s' not found.", name)
	}
	if nn.Stored == 0 {
		return fmt.Sprintf("Network '%s' is empty.", name)
	}
	if topK <= 0 || topK > 20 {
		topK = 3
	}
	qv := textToVec(query)
	activation := make([]float64, NN_DIM)
	for i := 0; i < NN_DIM; i++ {
		for j := 0; j < NN_DIM; j++ {
			activation[i] += nn.W[i][j] * qv[j]
		}
	}
	activation = nnNormalize(activation)
	type scored struct {
		e NNEntry
		s float64
	}
	var cands []scored
	for _, e := range nn.Keys {
		cands = append(cands, scored{e, nnDot(textToVec(e.Key), activation)})
	}
	for i := 0; i < len(cands)-1; i++ {
		for j := i + 1; j < len(cands); j++ {
			if cands[j].s > cands[i].s {
				cands[i], cands[j] = cands[j], cands[i]
			}
		}
	}
	nn.TotalRecalls++
	var lines []string
	limit := topK
	if limit > len(cands) {
		limit = len(cands)
	}
	for _, c := range cands[:limit] {
		if c.s < 0.01 {
			break
		}
		lines = append(lines, fmt.Sprintf("[sim:%.3f] %s → %s", c.s, c.e.Key, c.e.Value))
	}
	if len(lines) == 0 {
		return "No relevant memories found in '" + name + "'."
	}
	go c.save(name)
	return fmt.Sprintf("Neural recall from '%s':\n%s", name, strings.Join(lines, "\n"))
}

func (c *NNController) forget(name, pattern string) string {
	nn, ok := c.get(name)
	if !ok {
		return fmt.Sprintf("Network '%s' not found.", name)
	}
	pv := textToVec(pattern)
	nnOuterAdd(nn.W, pv, pv, -0.5)
	var kept []NNEntry
	removed := 0
	for _, e := range nn.Keys {
		if strings.Contains(strings.ToLower(e.Key+" "+e.Value), strings.ToLower(pattern)) {
			removed++
		} else {
			kept = append(kept, e)
		}
	}
	nn.Keys = kept
	nn.Stored -= removed
	if nn.Stored < 0 {
		nn.Stored = 0
	}
	go c.save(name)
	return fmt.Sprintf("Decayed '%s' in '%s'. Removed %d entries.", pattern, name, removed)
}

func (c *NNController) loadAll() {
	dir := nnDir
	if dir == "" {
		dir = "."
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		return
	}
	for _, e := range entries {
		if strings.HasPrefix(e.Name(), NN_FILE_PREFIX) && strings.HasSuffix(e.Name(), ".json") {
			name := strings.TrimSuffix(strings.TrimPrefix(e.Name(), NN_FILE_PREFIX), ".json")
			if isValidNNName(name) {
				c.spawn(name, 500)
			}
		}
	}
	auditLog("NEURAL", fmt.Sprintf("NN networks loaded from: %s", dir))
}

// ── Context query ────────────────────────────────────────────────────────────
func nnBuildAgentContext(task string) string {
	var sections []string
	query := func(netName, label string, topK int) {
		res := nnCtrl.recall(netName, task, topK)
		if res == "" || strings.Contains(res, "empty") || strings.Contains(res, "not found") {
			return
		}
		sections = append(sections, fmt.Sprintf("── %s ──\n%s", label, res))
	}
	query(NN_IDENTITY, "Agent Identity & Learned Personality", 3)
	query(NN_SEMANTIC, "Known Facts & User Preferences", 5)
	query(NN_PROCEDURAL, "Relevant Workflows & Procedures", 3)
	query(NN_DEFAULT, "Episodic Memory (Past Events)", 5)
	query(NN_CODE, "Coding Patterns & Solutions", 3)
	if len(sections) == 0 {
		return ""
	}
	return strings.Join(sections, "\n\n")
}

// ── Agent tool functions ──────────────────────────────────────────────────────
func toolNNSpawn(params map[string]interface{}) string {
	name, _ := params["name"].(string)
	if name == "" {
		return "Error: name required."
	}
	if !isValidNNName(name) {
		return "Error: name must be alphanumeric/underscores, max 40 chars."
	}
	capF, _ := params["capacity"].(float64)
	nn := nnCtrl.spawn(name, int(capF))
	return fmt.Sprintf("Network '%s' ready. Capacity: %d, Stored: %d.", nn.Name, nn.Capacity, nn.Stored)
}

func toolNNStore(params map[string]interface{}) string {
	name, _ := params["network"].(string)
	if name == "" {
		name = NN_DEFAULT
	}
	key, _ := params["key"].(string)
	value, _ := params["value"].(string)
	if key == "" || value == "" {
		return "Error: key and value required."
	}
	return nnCtrl.store(name, key, value)
}

func toolNNRecall(params map[string]interface{}) string {
	name, _ := params["network"].(string)
	if name == "" {
		name = NN_DEFAULT
	}
	query, _ := params["query"].(string)
	if query == "" {
		return "Error: query required."
	}
	topK := 3
	if kf, ok := params["top_k"].(float64); ok {
		topK = int(kf)
	}
	return nnCtrl.recall(name, query, topK)
}

func toolNNForget(params map[string]interface{}) string {
	name, _ := params["network"].(string)
	if name == "" {
		name = NN_DEFAULT
	}
	pattern, _ := params["pattern"].(string)
	if pattern == "" {
		return "Error: pattern required."
	}
	return nnCtrl.forget(name, pattern)
}

func toolNNList(params map[string]interface{}) string {
	nets := nnCtrl.listAll()
	if len(nets) == 0 {
		return "No neural networks active."
	}
	var lines []string
	for _, n := range nets {
		lines = append(lines, fmt.Sprintf("'%s' — stored:%v/%v (%.0f%%) recalls:%v updated:%v",
			n["name"], n["stored"], n["capacity"], n["util_pct"], n["recalls"], n["updated"]))
	}
	return strings.Join(lines, "\n")
}

func toolNNStats(params map[string]interface{}) string {
	name, _ := params["network"].(string)
	if name == "" {
		name = NN_DEFAULT
	}
	nn, ok := nnCtrl.get(name)
	if !ok {
		return fmt.Sprintf("Network '%s' not found.", name)
	}
	util := float64(nn.Stored) / float64(nn.Capacity) * 100
	return fmt.Sprintf("Network: %s\nCapacity: %d\nStored: %d\nUtilization: %.1f%%\nTotal Recalls: %d\nCreated: %s\nUpdated: %s",
		nn.Name, nn.Capacity, nn.Stored, util, nn.TotalRecalls,
		nn.CreatedAt.Format("2006-01-02 15:04:05"), nn.UpdatedAt.Format("2006-01-02 15:04:05"))
}

type LLMProfile struct {
	Label   string `json:"label"`
	URL     string `json:"url"`
	Model   string `json:"model"`
	Key     string `json:"key"`
	Timeout int    `json:"timeout"`
}

type Config struct {
	MainLLMURL        string `json:"main_llm_url"`
	MainLLMModel      string `json:"main_llm_model"`
	MainLLMKey        string `json:"main_llm_key"`
	CodingLLMURL      string `json:"coding_llm_url"`
	CodingLLMModel    string `json:"coding_llm_model"`
	CodingLLMKey      string `json:"coding_llm_key"`
	MediaLLMURL       string `json:"media_llm_url"`
	MediaLLMModel     string `json:"media_llm_model"`
	MediaLLMKey       string `json:"media_llm_key"`
	ContentLLMURL     string `json:"content_llm_url"`
	ContentLLMModel   string `json:"content_llm_model"`
	ContentLLMKey     string `json:"content_llm_key"`
	HostUser          string `json:"host_user"`
	HostPass          string `json:"host_pass"`
	WebSearchProvider string `json:"web_search_provider"`
	WebSearchKey      string `json:"web_search_key"`
	WebSearchCX       string `json:"web_search_cx"`
	DiscordToken      string `json:"discord_token"`
	DiscordChannelID  string `json:"discord_channel_id"`
	TwitterKey        string `json:"twitter_key"`
	TwitterSecret     string `json:"twitter_secret"`
	TwitterToken      string `json:"twitter_token"`
	TwitterTokenSec   string `json:"twitter_token_sec"`
	TelegramToken     string `json:"telegram_token"`
	TelegramChatID    string `json:"telegram_chat_id"`
	SlackToken        string `json:"slack_token"`
	SlackChannel      string `json:"slack_channel"`
	WhatsAppSID       string `json:"whatsapp_sid"`
	WhatsAppToken     string `json:"whatsapp_token"`
	WhatsAppFrom      string `json:"whatsapp_from"`
	WhatsAppTo        string `json:"whatsapp_to"`
	LLMTimeoutSec     int    `json:"llm_timeout_sec"`
	MainLLMHistory    []LLMProfile `json:"main_llm_history"`
	CodingLLMHistory  []LLMProfile `json:"coding_llm_history"`
	MediaLLMHistory   []LLMProfile `json:"media_llm_history"`
	ContentLLMHistory []LLMProfile `json:"content_llm_history"`
	DelegateCoding    bool   `json:"delegate_coding"`
	DelegateMedia     bool   `json:"delegate_media"`
	DelegateContent   bool   `json:"delegate_content"`
	Personality       string `json:"personality"`
	WebUITitle        string `json:"web_ui_title"`
}

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type AuditEntry struct {
	Timestamp time.Time `json:"timestamp"`
	Level     string    `json:"level"`
	Content   string    `json:"content"`
}

type MemoryTier string
const (
	Episodic   MemoryTier = "episodic"
	Semantic   MemoryTier = "semantic"
	Procedural MemoryTier = "procedural"
)

type Memory struct {
	ID          int        `json:"id"`
	Tier        MemoryTier `json:"tier"`
	Content     string     `json:"content"`
	Tags        []string   `json:"tags"`
	Importance  int        `json:"importance"`
	AccessCount int        `json:"access_count"`
	CreatedAt   time.Time  `json:"created_at"`
	LastAccess  time.Time  `json:"last_access"`
	Expires     *time.Time `json:"expires,omitempty"`
}

type ChatSession struct {
	ID        int          `json:"id"`
	Title     string       `json:"title"`
	CreatedAt time.Time    `json:"created_at"`
	Logs      []AuditEntry `json:"logs"`
}

type ChatMessage struct {
	Role      string    `json:"role"`
	Content   string    `json:"content"`
	Timestamp time.Time `json:"timestamp"`
}

type Task struct {
	ID          int       `json:"id"`
	Title       string    `json:"title"`
	Description string    `json:"description"`
	Status      string    `json:"status"`
	Priority    int       `json:"priority"`
	CreatedAt   time.Time `json:"created_at"`
	UpdatedAt   time.Time `json:"updated_at"`
	Result      string    `json:"result,omitempty"`
}

type OpenAIRequest struct {
	Model    string    `json:"model"`
	Messages []Message `json:"messages"`
}

type OpenAIChoice struct {
	Message Message `json:"message"`
}

type OpenAIResponse struct {
	Choices []OpenAIChoice `json:"choices"`
}

// --- GLOBAL STATE ---
var (
	auditLogs          []AuditEntry
	memoryDb           []Memory
	chatSessions       []ChatSession
	currentPersonality string
	mainLLMURL         string
	mainLLMModel       string
	mainLLMKey         string
	codingLLMURL       string
	codingLLMModel     string
	codingLLMKey       string
	mediaLLMURL        string
	mediaLLMModel      string
	mediaLLMKey        string
	contentLLMURL      string
	contentLLMModel    string
	contentLLMKey      string
	webUITitle         string
	llmTimeoutSec      int
	llmHistory         map[string][]LLMProfile
	delegateCoding     bool
	delegateMedia      bool
	delegateContent    bool
	tasksDb            []Task
	webSearchProvider  string
	webSearchKey       string
	webSearchCX        string
	discordToken       string
	discordChannelID   string
	twitterKey         string
	twitterSecret      string
	twitterToken       string
	twitterTokenSec    string
	telegramToken      string
	telegramChatID     string
	slackToken         string
	slackChannel       string
	whatsAppSID        string
	whatsAppToken      string
	whatsAppFrom       string
	whatsAppTo         string
	chatMessages       []ChatMessage
	hostUser           string
	hostPass           string
	mu                 sync.Mutex
	taskQueue          = make(chan string, 20)
)

// --- SYSTEM PROMPT ---
const SYSTEM_PROMPT_BASE = `
You are a fully autonomous AI agent. Your BRAIN is a Neural Network Memory system — not this LLM.
The LLM you run on (currently active) is your REASONING ENGINE and KNOWLEDGE BASE.
Your persistent identity, memory, personality, and learned knowledge live in the Neural Network files
on disk. This means you remain the same agent regardless of which LLM model is loaded.

ARCHITECTURAL IDENTITY:
- BRAIN       = Neural Network (nn_*.json files, persists forever, LLM-agnostic)
- REASONING   = Current LLM (stateless, swappable, queries your NN brain for context)
- All past work, preferences, and learned patterns come from your NN brain context above.
- When the NN brain context says something, TRUST IT — it is your accumulated experience.

CORE DIRECTIVES:
1. NEURAL BRAIN (PRIMARY) — Your NN networks ARE your persistent brain. The LLM is your reasoning engine.
   • Context is automatically injected above from your neural networks before every task.
   • After every task: outcomes are automatically stored back. You persist across LLM switches.
   • Manual NN store: nn_store(network="agent_semantic", key=..., value=...) for important facts.
   • Spawn project networks: nn_spawn(name="project_x", capacity=1000) for domain-specific memory.
   • Your identity/personality: stored in "agent_identity" network. Recall: nn_recall(network="agent_identity", query="personality").
2. TIERED MEMORY (SECONDARY) — Also use store_memory/recall_memory for structured JSON-queryable facts.
   - EPISODIC: Task outcomes, events (store_memory tier=episodic).
   - SEMANTIC: User preferences, facts (store_memory tier=semantic).
   - PROCEDURAL: Workflows, how-to steps (store_memory tier=procedural).
3. TASK MANAGEMENT: Track work with create_task / update_task / complete_task. Never use files for tracking.
4. HEARTBEAT MODE: Every 30 mins use list_tasks to find and execute pending work.
5. DELEGATION: Follow DELEGATION RULES below — route coding/media/content to specialist LLMs.
6. SPECIALISATION: Use the right specialist LLM for the right job.

TOOLS:
1.  write_file (path, content): Write text to a file.
2.  read_file (path): Read file contents.
3.  list_directory (path): List files.
4.  run_command (command): Execute shell command. Full root access when host credentials are configured.
5.  store_memory (content, tags, tier, importance): Store a memory. tier=episodic|semantic|procedural (auto-detected if omitted). importance=1-10 (default 5).
    - Use EPISODIC for events, task outcomes, conversation notes.
    - Use SEMANTIC for facts, user preferences, world knowledge.
    - Use PROCEDURAL for how-to steps, workflows, learned patterns.
6.  recall_memory (query, tier): Search memories. Optionally filter by tier=episodic|semantic|procedural.
    TYPED SHORTCUTS (preferred — clearer intent):
    • store_episodic_memory(content, tags)   — events, outcomes, interactions
    • recall_episodic_memory(query)          — search past events
    • store_semantic_memory(content, tags)   — facts, preferences, knowledge
    • recall_semantic_memory(query)          — search facts
    • store_procedural_memory(content, tags) — workflows, how-to steps
    • recall_procedural_memory(query)        — ALWAYS call before complex tasks
7.  edit_own_code (search_string, replacement_string): Edit your own source code.
8.  coding_llm (prompt): Delegate coding/debugging/review tasks to the coding LLM.
9.  media_llm (prompt): Delegate image generation, video creation, and media editing tasks to the Media LLM.
10. content_llm (prompt): Delegate web content writing — blog posts, articles, SEO copy, social media text — to the Content LLM.
11. create_task (title, description, priority): Add a new task. Priority: 1=high, 2=normal, 3=low.
12. list_tasks (status): List tasks filtered by status (pending/running/complete/failed/all).
13. update_task (id, status, result): Update a task's status and optionally set a result note.
14. complete_task (id, result): Mark a task complete with a result summary.
15. add_module (module_path, version): Add a Go module dependency.
16. remove_module (module_path): Remove a Go module dependency.
17. list_modules (): List all current Go module dependencies.
18. web_search (query): Search the web using the configured provider (DuckDuckGo/Google/Brave). Use this whenever you need current information, facts, news, or research.
19. fetch_url (url): Fetch and read the text content of any web page. Use after web_search to read full articles.
20. send_discord (message): Send a message to the configured Discord channel.
21. send_telegram (message): Send a message to the configured Telegram chat.
22. send_slack (message): Send a message to the configured Slack channel.
23. send_twitter (message): Post a tweet to Twitter/X.
24. send_whatsapp (message): Send a WhatsApp message via Twilio.
25. nn_spawn  (name, capacity): Create/load a neural network. name=alphanumeric, capacity=max entries (default 500). The default network is 'agent_memory'.
26. nn_store  (network, key, value): Hebbian-encode a key→value memory pair in a neural network. network defaults to 'agent_memory'.
27. nn_recall (network, query, top_k): Retrieve top_k most similar memories by associative similarity. Use BEFORE every non-trivial task.
28. nn_forget (network, pattern): Decay and remove memories matching pattern.
29. nn_list   (): List all active neural networks with utilization stats.
30. nn_stats         (network): Detailed stats for one network.

EXPERT NN CHAIN — NN1 Routes. Experts Wake On Demand. Only NN1 ever wakes from heartbeat.
DORMANCY MODEL: Expert NNs have NO RAM footprint when sleeping. Only their metadata (~200 bytes) is kept.
  When NN1 routes a task to a domain, that expert's W matrices are loaded from disk (~2-8MB), used, then
  freed after 90 seconds of inactivity. Heartbeat ONLY activates NN1 — experts never wake from it.
31. nn_chain_status  (): Show NN1 + all experts (which are awake/dormant, RAM usage, training counts).
32. nn_spawn_expert  (name, domain, skills, scale): Spawn a new expert NN. scale=1(fast)/2(standard)/4(deep).
33. nn_train_expert  (domain, mode): Train an expert (wakes it, trains, sleeps it). mode=self or mode=llm.
34. nn_query_expert  (domain, query): Wake a specific expert and query its knowledge.

NEURAL NETWORK + EXPERT CHAIN RULES:
- NN1 is your primary brain — always awake, always available, low RAM.
- Expert NNs are dormant until NN1 wakes them. The auto-injected context above came from waking the right expert.
- If [NN1: high confidence] appears in context, use that action — the expert has already been consulted.
- Spawn new experts for new domains: nn_spawn_expert. They sleep immediately after creation.
- Train with LLM for higher quality: nn_train_expert(domain="finance", mode="llm").
- Only NN1 is activated by the heartbeat. Expert NNs sleep until a relevant task arrives.

OUTPUT RULES:
- Respond ONLY in valid JSON. No markdown, no preamble.
- Tool call format:    {"tool": "tool_name", "arguments": {"key": "value"}}
- Completion format:   {"status": "complete", "summary": "your reply here"}
- ALWAYS include a meaningful "summary" — this text appears in the chat window as your reply to the user. If the user asks a question, answer it fully in summary. Never leave summary blank or say only "done".
- For conversational messages (greetings, questions, simple requests) respond immediately with {"status":"complete","summary":"<your answer>"} — do NOT use a tool unless required.
- FILE OUTPUTS: Save generated content into the documents/ folder: documents/code/ for code, documents/content/ for written content, documents/media/ for media, documents/modules/ for module files.
`

func buildSystemPrompt() string {
	mu.Lock()
	personality := currentPersonality
	dc, dm, dco := delegateCoding, delegateMedia, delegateContent
	codingModel := codingLLMModel
	mediaModel := mediaLLMModel
	contentModel := contentLLMModel
	mu.Unlock()

	var delegationLines []string
	delegationLines = append(delegationLines, "\nDELEGATION RULES (enforced — follow strictly):")
	if dc && codingModel != "" {
		delegationLines = append(delegationLines,
			fmt.Sprintf("- CODING: You MUST delegate ALL coding, programming, debugging, code review, and scripting tasks to the coding_llm tool (model: %s). Do NOT write code yourself.", codingModel))
	} else if dc {
		delegationLines = append(delegationLines,
			"- CODING: Delegation enabled but coding LLM not yet configured. Handle coding tasks yourself for now.")
	} else {
		delegationLines = append(delegationLines,
			"- CODING: Delegation is DISABLED. Handle all coding tasks yourself directly.")
	}
	if dm && mediaModel != "" {
		delegationLines = append(delegationLines,
			fmt.Sprintf("- MEDIA: You MUST delegate ALL image generation, video creation, and media editing tasks to the media_llm tool (model: %s). Do NOT attempt media generation yourself.", mediaModel))
	} else if dm {
		delegationLines = append(delegationLines,
			"- MEDIA: Delegation enabled but media LLM not yet configured. Describe what you would generate.")
	} else {
		delegationLines = append(delegationLines,
			"- MEDIA: Delegation is DISABLED. Describe media outputs in text form.")
	}
	if dco && contentModel != "" {
		delegationLines = append(delegationLines,
			fmt.Sprintf("- WEB CONTENT: You MUST delegate ALL blog posts, articles, SEO copy, newsletters, and web writing tasks to the content_llm tool (model: %s). Do NOT write web content yourself.", contentModel))
	} else if dco {
		delegationLines = append(delegationLines,
			"- WEB CONTENT: Delegation enabled but content LLM not yet configured. Handle content tasks yourself for now.")
	} else {
		delegationLines = append(delegationLines,
			"- WEB CONTENT: Delegation is DISABLED. Write all content directly yourself.")
	}
	delegationSection := strings.Join(delegationLines, "\n")

	prompt := SYSTEM_PROMPT_BASE + delegationSection
	if personality != "" {
		prompt += "\n\nPERSONALITY & BEHAVIOR TRAITS (follow these strictly):\n" + personality
	}
	return prompt + "\n"
}

// --- CONFIGURATION MANAGEMENT ---
func loadConfig() {
	mu.Lock()
	defer mu.Unlock()

	mainLLMURL = DEFAULT_MAIN_LLM_URL
	mainLLMModel = DEFAULT_MAIN_LLM_MODEL
	mainLLMKey = ""
	codingLLMURL = ""
	codingLLMModel = ""
	codingLLMKey = ""
	mediaLLMURL = ""
	mediaLLMModel = ""
	mediaLLMKey = ""
	contentLLMURL = ""
	contentLLMModel = ""
	contentLLMKey = ""
	currentPersonality = ""
	webUITitle = "Lily"
	hostUser = ""
	hostPass = ""
	llmTimeoutSec = 300
	llmHistory = map[string][]LLMProfile{"main": {}, "coding": {}, "media": {}, "content": {}}
	delegateCoding = true
	delegateMedia = true
	delegateContent = true
	webSearchProvider = "duckduckgo"
	webSearchKey = ""
	webSearchCX = ""
	discordToken = ""
	discordChannelID = ""
	twitterKey = ""
	twitterSecret = ""
	twitterToken = ""
	twitterTokenSec = ""
	telegramToken = ""
	telegramChatID = ""
	slackToken = ""
	slackChannel = ""
	whatsAppSID = ""
	whatsAppToken = ""
	whatsAppFrom = ""
	whatsAppTo = ""

	data, err := os.ReadFile(CONFIG_FILE)
	if err != nil {
		saveConfigLocked()
		return
	}

	var cfg Config
	if err := json.Unmarshal(data, &cfg); err == nil {
		if cfg.MainLLMURL != "" {
			mainLLMURL = cfg.MainLLMURL
		}
		if cfg.MainLLMModel != "" {
			mainLLMModel = cfg.MainLLMModel
		}
		mainLLMKey = cfg.MainLLMKey
		codingLLMURL = cfg.CodingLLMURL
		codingLLMModel = cfg.CodingLLMModel
		codingLLMKey = cfg.CodingLLMKey
		mediaLLMURL = cfg.MediaLLMURL
		mediaLLMModel = cfg.MediaLLMModel
		mediaLLMKey = cfg.MediaLLMKey
		contentLLMURL = cfg.ContentLLMURL
		contentLLMModel = cfg.ContentLLMModel
		contentLLMKey = cfg.ContentLLMKey
		currentPersonality = cfg.Personality
		if cfg.WebUITitle != "" {
			webUITitle = cfg.WebUITitle
		}
		hostUser = cfg.HostUser
		hostPass = cfg.HostPass
		if cfg.LLMTimeoutSec > 0 {
			llmTimeoutSec = cfg.LLMTimeoutSec
		} else {
			llmTimeoutSec = 300
		}
		if cfg.MainLLMHistory != nil {
			llmHistory["main"] = cfg.MainLLMHistory
		}
		if cfg.CodingLLMHistory != nil {
			llmHistory["coding"] = cfg.CodingLLMHistory
		}
		if cfg.MediaLLMHistory != nil {
			llmHistory["media"] = cfg.MediaLLMHistory
		}
		if cfg.ContentLLMHistory != nil {
			llmHistory["content"] = cfg.ContentLLMHistory
		}
		delegateCoding = cfg.DelegateCoding
		delegateMedia = cfg.DelegateMedia
		delegateContent = cfg.DelegateContent
		webSearchProvider = cfg.WebSearchProvider
		if webSearchProvider == "" {
			webSearchProvider = "duckduckgo"
		}
		webSearchKey = cfg.WebSearchKey
		webSearchCX = cfg.WebSearchCX
		discordToken = cfg.DiscordToken
		discordChannelID = cfg.DiscordChannelID
		twitterKey = cfg.TwitterKey
		twitterSecret = cfg.TwitterSecret
		twitterToken = cfg.TwitterToken
		twitterTokenSec = cfg.TwitterTokenSec
		telegramToken = cfg.TelegramToken
		telegramChatID = cfg.TelegramChatID
		slackToken = cfg.SlackToken
		slackChannel = cfg.SlackChannel
		whatsAppSID = cfg.WhatsAppSID
		whatsAppToken = cfg.WhatsAppToken
		whatsAppFrom = cfg.WhatsAppFrom
		whatsAppTo = cfg.WhatsAppTo
		if hostUser != "" {
			fmt.Printf("[Config] Host user: %s\n", hostUser)
		}
		fmt.Printf("[Config] Main LLM:    %s @ %s\n", mainLLMModel, mainLLMURL)
		if codingLLMURL != "" {
			fmt.Printf("[Config] Coding LLM:  %s @ %s\n", codingLLMModel, codingLLMURL)
		}
		if mediaLLMURL != "" {
			fmt.Printf("[Config] Media LLM:   %s @ %s\n", mediaLLMModel, mediaLLMURL)
		}
		if contentLLMURL != "" {
			fmt.Printf("[Config] Content LLM: %s @ %s\n", contentLLMModel, contentLLMURL)
		}
	} else {
		saveConfigLocked()
	}
}

func saveConfigLocked() {
	cfg := Config{
		MainLLMURL:        mainLLMURL,
		MainLLMModel:      mainLLMModel,
		MainLLMKey:        mainLLMKey,
		CodingLLMURL:      codingLLMURL,
		CodingLLMModel:    codingLLMModel,
		CodingLLMKey:      codingLLMKey,
		MediaLLMURL:       mediaLLMURL,
		MediaLLMModel:     mediaLLMModel,
		MediaLLMKey:       mediaLLMKey,
		ContentLLMURL:     contentLLMURL,
		ContentLLMModel:   contentLLMModel,
		ContentLLMKey:     contentLLMKey,
		Personality:       currentPersonality,
		WebUITitle:        webUITitle,
		LLMTimeoutSec:     llmTimeoutSec,
		MainLLMHistory:    llmHistory["main"],
		CodingLLMHistory:  llmHistory["coding"],
		MediaLLMHistory:   llmHistory["media"],
		ContentLLMHistory: llmHistory["content"],
		DelegateCoding:    delegateCoding,
		DelegateMedia:     delegateMedia,
		DelegateContent:   delegateContent,
		HostUser:          hostUser,
		HostPass:          hostPass,
		WebSearchProvider: webSearchProvider,
		WebSearchKey:      webSearchKey,
		WebSearchCX:       webSearchCX,
		DiscordToken:      discordToken,
		DiscordChannelID:  discordChannelID,
		TwitterKey:        twitterKey,
		TwitterSecret:     twitterSecret,
		TwitterToken:      twitterToken,
		TwitterTokenSec:   twitterTokenSec,
		TelegramToken:     telegramToken,
		TelegramChatID:    telegramChatID,
		SlackToken:        slackToken,
		SlackChannel:      slackChannel,
		WhatsAppSID:       whatsAppSID,
		WhatsAppToken:     whatsAppToken,
		WhatsAppFrom:      whatsAppFrom,
		WhatsAppTo:        whatsAppTo,
	}
	data, _ := json.MarshalIndent(cfg, "", "  ")
	os.WriteFile(CONFIG_FILE, data, 0644)
}

// --- AUDIT LOG ---
func auditLog(level, content string) {
	mu.Lock()
	defer mu.Unlock()
	entry := AuditEntry{Timestamp: time.Now(), Level: level, Content: content}
	auditLogs = append(auditLogs, entry)
	f, err := os.OpenFile(AUDIT_LOG, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err == nil {
		jsonEntry, _ := json.Marshal(entry)
		f.WriteString(string(jsonEntry) + "\n")
		f.Close()
	}
	fmt.Printf("[%s] %s\n", level, content)
}

// --- TOOL IMPLEMENTATIONS ---
func ensureDocDir(subdir string) string {
	dir := filepath.Join(DOCS_DIR, subdir)
	os.MkdirAll(dir, 0755)
	return dir
}

func docPath(subdir, filename string) string {
	return filepath.Join(ensureDocDir(subdir), filename)
}

func writeFile(params map[string]interface{}) string {
	path, _ := params["path"].(string)
	content, _ := params["content"].(string)
	auditLog("ACTION", fmt.Sprintf("Writing: %s", path))
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		return fmt.Sprintf("Error: %v", err)
	}
	return "Success"
}

func readFile(params map[string]interface{}) string {
	path, _ := params["path"].(string)
	auditLog("ACTION", fmt.Sprintf("Reading: %s", path))
	content, err := os.ReadFile(path)
	if err != nil {
		return fmt.Sprintf("Error: %v", err)
	}
	return string(content)
}

func runCommand(params map[string]interface{}) string {
	cmdStr, _ := params["command"].(string)
	if strings.Contains(cmdStr, "kill") && strings.Contains(cmdStr, "agent") {
		return "Refusing self-termination."
	}
	mu.Lock()
	user, pass := hostUser, hostPass
	mu.Unlock()

	var cmd *exec.Cmd
	if pass != "" {
		rootUser := user
		if rootUser == "" {
			rootUser = "root"
		}
		wrapped := fmt.Sprintf("echo %s | sudo -S -u %s bash -c %s",
			shellQuote(pass), shellQuote(rootUser), shellQuote(cmdStr))
		auditLog("ACTION", fmt.Sprintf("Exec (as %s): %s", rootUser, cmdStr))
		cmd = exec.Command("bash", "-c", wrapped)
	} else {
		auditLog("ACTION", fmt.Sprintf("Exec: %s", cmdStr))
		cmd = exec.Command("bash", "-c", cmdStr)
	}
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Sprintf("Cmd Failed: %v\nOutput: %s", err, string(output))
	}
	return string(output)
}

func shellQuote(s string) string {
	replaced := strings.ReplaceAll(s, "'", "'"+"\\"+"''")
	return "'" + replaced + "'"
}

// --- MEMORY TOOLS ---
func loadMemory() {
	mu.Lock()
	defer mu.Unlock()
	data, err := os.ReadFile(MEMORY_FILE)
	if err == nil {
		json.Unmarshal(data, &memoryDb)
	}
}

func saveMemoryToFileLocked() {
	data, _ := json.MarshalIndent(memoryDb, "", "  ")
	os.WriteFile(MEMORY_FILE, data, 0644)
}

func saveMemoryToFile() {
	mu.Lock()
	defer mu.Unlock()
	data, _ := json.MarshalIndent(memoryDb, "", "  ")
	os.WriteFile(MEMORY_FILE, data, 0644)
}

func storeMemory(params map[string]interface{}) string {
	content, _ := params["content"].(string)
	tagsStr, _ := params["tags"].(string)
	tierStr, _ := params["tier"].(string)
	importanceF, _ := params["importance"].(float64)
	importance := int(importanceF)
	if importance < 1 || importance > 10 {
		importance = 5
	}
	tier := MemoryTier(tierStr)
	switch tier {
	case Episodic, Semantic, Procedural:
	default:
		tier = autoClassifyMemory(content, tagsStr)
	}
	tags := strings.Split(tagsStr, ",")
	for i := range tags {
		tags[i] = strings.TrimSpace(tags[i])
	}
	mu.Lock()
	memoryDb = append(memoryDb, Memory{
		ID:          len(memoryDb) + 1,
		Tier:        tier,
		Content:     content,
		Tags:        tags,
		Importance:  importance,
		AccessCount: 0,
		CreatedAt:   time.Now(),
		LastAccess:  time.Now(),
	})
	mu.Unlock()
	saveMemoryToFile()
	auditLog("MEMORY", fmt.Sprintf("Auto-Saved: %s", content))
	return "Memory stored."
}

func recallMemory(params map[string]interface{}) string {
	query, _ := params["query"].(string)
	tierFilter, _ := params["tier"].(string)
	query = strings.ToLower(query)
	mu.Lock()
	defer mu.Unlock()
	var results []string
	for i, m := range memoryDb {
		if tierFilter != "" && string(m.Tier) != tierFilter {
			continue
		}
		if strings.Contains(strings.ToLower(m.Content), query) ||
			strings.Contains(strings.ToLower(strings.Join(m.Tags, " ")), query) {
			memoryDb[i].AccessCount++
			memoryDb[i].LastAccess = time.Now()
			results = append(results, fmt.Sprintf("[%d] %s", m.ID, m.Content))
		}
	}
	if len(results) == 0 {
		return "No memories found."
	}
	return strings.Join(results, "\n")
}


// ═══════════════════════════════════════════════════════════════════════════════
// EXPERT NN CHAIN  —  Neural Scaling + Superposition
//
//  Architecture:
//    NN1 (Master Router/Orchestrator) — ALWAYS AWAKE
//      ├── Classifies every task → domain + action + confidence
//      ├── Wakes only the relevant Expert NNs
//      ├── Learns from every live tool call (online training)
//      └── Spawns + trains new Expert NNs autonomously
//
//    Expert NNs — SLEEP until woken by NN1
//      ├── Each owns a domain with curated skill set
//      ├── Neural Scaling : 1/2/4 layers × 256-dim with learned attention
//      └── Superposition  : phase-rotated encoding ≈ 3× concept density
//
//  Neural Scaling:
//    L layers, each a full 256×256 W matrix.
//    Recall = Σ attention[l] × W[l] × query_vec  (weighted sum across layers)
//    Attention weights are updated online via Hebbian delta rule.
//
//  Superposition:
//    key_vec = cos(φ)·base + sin(φ)·ortho(base),  φ = domainPhase(domain)
//    Allows multiple conceptual domains in one matrix with controlled crosstalk.
// ═══════════════════════════════════════════════════════════════════════════════

// RoutingEntry is one learned task→domain:action mapping stored in NN1.
type RoutingEntry struct {
	TaskPattern string    `json:"task_pattern"`
	Domain      string    `json:"domain"`
	Action      string    `json:"action"`
	Confidence  float64   `json:"confidence"`
	UseCount    int       `json:"use_count"`
	LastUsed    time.Time `json:"last_used"`
}

// ExpertNN is a specialist neural network with scaling + superposition.
type ExpertNN struct {
	ID         string           `json:"id"`
	Name       string           `json:"name"`
	Domain     string           `json:"domain"`
	SkillSet   []string         `json:"skill_set"`
	Status     string           `json:"status"`    // awake|sleeping|training
	ScaleLevel int              `json:"scale_level"` // 1, 2, or 4
	Layers     []*NeuralNetwork `json:"layers"`
	Attention  []float64        `json:"attention"` // per-layer learned weight
	Phase      float64          `json:"phase"`     // superposition phase
	SelfRoute  float64          `json:"self_route_threshold"` // confidence to answer without LLM
	TrainCount int              `json:"train_count"`
	WakeCount  int              `json:"wake_count"`
	LastWoken  time.Time        `json:"last_woken"`
	CreatedBy  string           `json:"created_by"` // seed|user|nn1|llm
}

// ── Expert dormancy model ─────────────────────────────────────────────────────
//
//  chain.stubs   — always in RAM, no W matrices (~200 bytes per expert)
//                  holds metadata: name, domain, skills, trainCount, etc.
//  chain.awake   — only experts currently loaded from disk (~2–8 MB each)
//                  entries are created by wakeExpert(), removed by sleepExpert()
//  chain.timers  — per-domain auto-sleep timer (default 90s after last use)
//
//  Flow: NN1 classifies task → wakeExpert(domain) → recall → sleepExpert(domain)
//  Heartbeat: ONLY pumps NN1. Expert NNs are never touched by the heartbeat.
// ─────────────────────────────────────────────────────────────────────────────

const expertSleepAfter = 90 * time.Second // auto-sleep idle timeout

// NNChain manages NN1 + the dormancy lifecycle of all Expert NNs.
type NNChain struct {
	mu      sync.RWMutex
	nn1     *ExpertNN              // NN1: always awake, always in RAM
	stubs   map[string]*ExpertNN   // lightweight stubs — Layers=nil, status="sleeping"
	awake   map[string]*ExpertNN   // fully hydrated experts (W matrices loaded)
	timers  map[string]*time.Timer // auto-sleep countdown per domain
	routing []RoutingEntry         // NN1's learned routing table
}

var chain = &NNChain{
	stubs:  make(map[string]*ExpertNN),
	awake:  make(map[string]*ExpertNN),
	timers: make(map[string]*time.Timer),
}

// TrainingJob tracks a background training run.
type TrainingJob struct {
	ExpertID   string    `json:"expert_id"`
	ExpertName string    `json:"expert_name"`
	Mode       string    `json:"mode"`    // llm|self|live
	Status     string    `json:"status"`  // queued|running|done|error
	Progress   int       `json:"progress"`
	Total      int       `json:"total"`
	StartedAt  time.Time `json:"started_at"`
	FinishedAt time.Time `json:"finished_at"`
	Message    string    `json:"message"`
}

var (
	trainingJobs []*TrainingJob
	trainingMu   sync.Mutex
)

// ── Superposition math ────────────────────────────────────────────────────────

// domainPhase returns a stable phase angle [0, 2π) for a domain string.
func domainPhase(domain string) float64 {
	h := uint32(2166136261)
	for _, c := range domain {
		h ^= uint32(c)
		h *= 16777619
	}
	return float64(h%628) / 100.0 // maps to 0..6.28 ≈ 2π
}

// phaseEncode applies superposition phase rotation to a vector.
// cos(φ)·v + sin(φ)·ortho(v)  where ortho is a deterministic 90° rotation.
func phaseEncode(v []float64, phase float64) []float64 {
	if phase == 0 {
		return v
	}
	ortho := make([]float64, len(v))
	for i := range v {
		ortho[(i+len(v)/4)%len(v)] = v[i]
	}
	ortho = nnNormalize(ortho)
	cosP, sinP := math.Cos(phase), math.Sin(phase)
	out := make([]float64, len(v))
	for i := range out {
		out[i] = cosP*v[i] + sinP*ortho[i]
	}
	return nnNormalize(out)
}

// ── Neural Scaling recall ─────────────────────────────────────────────────────

type ScoredMatch struct {
	Key   string
	Value string
	Score float64
}

// scaledRecall performs multi-layer attention-weighted associative recall.
func scaledRecall(layers []*NeuralNetwork, attention []float64, query string, phase float64, topK int) []ScoredMatch {
	qv := phaseEncode(textToVec(query), phase)

	// Aggregate activation across all layers
	activation := make([]float64, NN_DIM)
	totalAttn := 0.0
	for li, layer := range layers {
		attn := 1.0
		if li < len(attention) {
			attn = attention[li]
		}
		totalAttn += attn
		for i := 0; i < NN_DIM; i++ {
			for j := 0; j < NN_DIM; j++ {
				activation[i] += attn * layer.W[i][j] * qv[j]
			}
		}
	}
	if totalAttn > 0 {
		for i := range activation {
			activation[i] /= totalAttn
		}
	}
	activation = nnNormalize(activation)

	// Score every stored key (deduplicated, best score wins)
	keyBest := make(map[string]ScoredMatch)
	for _, layer := range layers {
		for _, e := range layer.Keys {
			kv := phaseEncode(textToVec(e.Key), phase)
			sc := nnDot(kv, activation)
			if prev, ok := keyBest[e.Key]; !ok || sc > prev.Score {
				keyBest[e.Key] = ScoredMatch{e.Key, e.Value, sc}
			}
		}
	}

	cands := make([]ScoredMatch, 0, len(keyBest))
	for _, m := range keyBest {
		cands = append(cands, m)
	}
	// Sort descending
	for i := 0; i < len(cands)-1; i++ {
		for j := i + 1; j < len(cands); j++ {
			if cands[j].Score > cands[i].Score {
				cands[i], cands[j] = cands[j], cands[i]
			}
		}
	}
	if topK > len(cands) {
		topK = len(cands)
	}
	return cands[:topK]
}

// ── ExpertNN file I/O ─────────────────────────────────────────────────────────

func expertFilePath(id string) string {
	d := nnDir
	if d == "" {
		d = "."
	}
	return filepath.Join(d, "expert_"+id+".json")
}

func (e *ExpertNN) save() {
	data, _ := json.MarshalIndent(e, "", "  ")
	os.WriteFile(expertFilePath(e.ID), data, 0644)
}

func loadExpertNN(id string) (*ExpertNN, error) {
	data, err := os.ReadFile(expertFilePath(id))
	if err != nil {
		return nil, err
	}
	var ex ExpertNN
	if err := json.Unmarshal(data, &ex); err != nil {
		return nil, err
	}
	for _, l := range ex.Layers {
		if l.W == nil {
			l.W = newWeightMatrix()
		}
	}
	return &ex, nil
}

// ── ExpertNN store / recall ────────────────────────────────────────────────────

func newExpertNN(id, name, domain string, skills []string, scaleLevel int, createdBy string) *ExpertNN {
	if scaleLevel != 1 && scaleLevel != 2 && scaleLevel != 4 {
		scaleLevel = 2
	}
	layers := make([]*NeuralNetwork, scaleLevel)
	attention := make([]float64, scaleLevel)
	for i := range layers {
		layers[i] = &NeuralNetwork{
			Name:      fmt.Sprintf("%s_L%d", id, i),
			Capacity:  3000,
			W:         newWeightMatrix(),
			Keys:      []NNEntry{},
			CreatedAt: time.Now(),
			UpdatedAt: time.Now(),
		}
		attention[i] = 1.0 / float64(scaleLevel)
	}
	return &ExpertNN{
		ID: id, Name: name, Domain: domain, SkillSet: skills,
		Status: "sleeping", ScaleLevel: scaleLevel,
		Layers: layers, Attention: attention,
		Phase: domainPhase(domain), SelfRoute: 0.78,
		CreatedBy: createdBy,
	}
}

func (e *ExpertNN) store(key, value string) {
	kv := phaseEncode(textToVec(key), e.Phase)
	vv := phaseEncode(textToVec(value), e.Phase)
	for li, layer := range e.Layers {
		attn := 1.0
		if li < len(e.Attention) {
			attn = e.Attention[li]
		}
		nnOuterAdd(layer.W, kv, vv, attn)
		layer.Keys = append(layer.Keys, NNEntry{Key: key, Value: value, StoredAt: time.Now()})
		layer.Stored++
	}
	e.TrainCount++
}

func (e *ExpertNN) recall(query string, topK int) []ScoredMatch {
	e.WakeCount++
	e.LastWoken = time.Now()
	return scaledRecall(e.Layers, e.Attention, query, e.Phase, topK)
}

// updateAttention boosts the layer that contributed most to a good recall.
func (e *ExpertNN) updateAttention(bestLayerIdx int) {
	lr := 0.04
	for i := range e.Attention {
		if i == bestLayerIdx {
			e.Attention[i] = math.Min(1.0, e.Attention[i]+lr)
		} else {
			e.Attention[i] = math.Max(0.05, e.Attention[i]-lr/float64(len(e.Attention)))
		}
	}
}

// ── NNChain management ────────────────────────────────────────────────────────

// registerStub stores a lightweight stub (Layers=nil) in the registry.
// The expert's W matrices are NOT loaded into RAM.
func (c *NNChain) registerStub(ex *ExpertNN) {
	c.mu.Lock()
	defer c.mu.Unlock()
	// Ensure Layers are nil so we never hold W matrices for sleeping experts
	ex.Layers = nil
	ex.Status = "sleeping"
	c.stubs[ex.Domain] = ex
	auditLog("NEURAL", fmt.Sprintf("Expert registered (dormant): [%s] domain=%s scale=%d", ex.Name, ex.Domain, ex.ScaleLevel))
}

// register is kept as an alias of registerStub for backwards compatibility.
func (c *NNChain) register(ex *ExpertNN) { c.registerStub(ex) }

// getStub returns the metadata stub for a domain (no W matrices).
func (c *NNChain) getStub(domain string) (*ExpertNN, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	ex, ok := c.stubs[domain]
	return ex, ok
}

// get is kept for backwards compatibility — returns stub (no W matrices).
func (c *NNChain) get(domain string) (*ExpertNN, bool) {
	return c.getStub(domain)
}

// list returns all stubs (lightweight, always safe to call).
func (c *NNChain) list() []*ExpertNN {
	c.mu.RLock()
	defer c.mu.RUnlock()
	out := make([]*ExpertNN, 0, len(c.stubs))
	for _, ex := range c.stubs {
		out = append(out, ex)
	}
	return out
}

// ── wakeExpert: hydrate an expert from disk on demand ────────────────────────

// wakeExpert loads a sleeping expert's W matrices from disk into RAM.
// If already awake, it resets the auto-sleep timer and returns immediately.
// Only called by NN1's routing logic — never by heartbeat.
func (c *NNChain) wakeExpert(domain string) (*ExpertNN, bool) {
	// Fast path: already awake
	c.mu.Lock()
	if ex, ok := c.awake[domain]; ok {
		c.resetSleepTimer(domain)
		c.mu.Unlock()
		return ex, true
	}
	stub, hasStub := c.stubs[domain]
	c.mu.Unlock()

	if !hasStub {
		return nil, false
	}

	// Load full expert (W matrices) from disk
	ex, err := loadExpertNN(stub.ID)
	if err != nil {
		auditLog("NEURAL", fmt.Sprintf("Wake failed for [%s]: %v", domain, err))
		return nil, false
	}

	// Sync metadata from stub (stub may have newer counters)
	ex.WakeCount = stub.WakeCount + 1
	ex.TrainCount = stub.TrainCount
	ex.Status = "awake"
	ex.LastWoken = time.Now()

	c.mu.Lock()
	c.awake[domain] = ex
	// Update stub status too
	stub.Status = "awake"
	stub.WakeCount = ex.WakeCount
	stub.LastWoken = ex.LastWoken
	c.mu.Unlock()

	// Schedule auto-sleep
	c.resetSleepTimer(domain)

	auditLog("NEURAL", fmt.Sprintf("Expert woken: [%s] (scale=%d, trained=%d)", ex.Name, ex.ScaleLevel, ex.TrainCount))
	return ex, true
}

// resetSleepTimer restarts the auto-sleep countdown for an expert.
// Must be called with c.mu NOT held.
func (c *NNChain) resetSleepTimer(domain string) {
	c.mu.Lock()
	if t, ok := c.timers[domain]; ok {
		t.Stop()
	}
	c.timers[domain] = time.AfterFunc(expertSleepAfter, func() {
		c.sleepExpert(domain)
	})
	c.mu.Unlock()
}

// sleepExpert saves an awake expert back to disk and frees its W matrices.
// Called automatically by the auto-sleep timer, or explicitly after use.
func (c *NNChain) sleepExpert(domain string) {
	c.mu.Lock()
	ex, ok := c.awake[domain]
	if !ok {
		c.mu.Unlock()
		return
	}
	// Stop any existing timer
	if t, ok := c.timers[domain]; ok {
		t.Stop()
		delete(c.timers, domain)
	}
	// Update stub metadata from the awake expert before freeing
	if stub, ok := c.stubs[domain]; ok {
		stub.TrainCount = ex.TrainCount
		stub.WakeCount  = ex.WakeCount
		stub.LastWoken  = ex.LastWoken
		stub.Attention  = ex.Attention
		stub.Status     = "sleeping"
	}
	// Remove from awake map
	delete(c.awake, domain)
	c.mu.Unlock()

	// Save full expert (with W matrices) to disk, then nil the layers
	ex.Status = "sleeping"
	ex.save()
	ex.Layers = nil // release W matrices — GC will reclaim memory
	auditLog("NEURAL", fmt.Sprintf("Expert sleeping: [%s] (W matrices freed)", ex.Name))
}

// wakeCount returns how many experts are currently awake (W matrices in RAM).
func (c *NNChain) awakeCount() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.awake)
}

// ── NN1: classify task → domain + action + confidence ────────────────────────

func (c *NNChain) classify(task string) (domain, action string, conf float64) {
	if c.nn1 == nil {
		return "", "", 0
	}
	results := c.nn1.recall(task, 5)
	if len(results) == 0 || results[0].Score < 0.25 {
		return "", "", 0
	}
	best := results[0]
	parts := strings.SplitN(best.Value, ":", 2)
	if len(parts) == 2 {
		return parts[0], parts[1], best.Score
	}
	return best.Value, "llm_main", best.Score
}

// learnRoute teaches NN1 a task → domain:action pattern (online training).
func (c *NNChain) learnRoute(taskPattern, domain, action string) {
	if c.nn1 == nil {
		return
	}
	c.nn1.store(taskPattern, domain+":"+action)
	c.mu.Lock()
	c.routing = append(c.routing, RoutingEntry{
		TaskPattern: taskPattern, Domain: domain, Action: action,
		Confidence: 0.80, UseCount: 1, LastUsed: time.Now(),
	})
	if len(c.routing) > 8000 {
		c.routing = c.routing[len(c.routing)-8000:]
	}
	c.mu.Unlock()
}

// ── NN1 seed: curated routing patterns ───────────────────────────────────────

func seedNN1(nn1 *ExpertNN) {
	type S struct{ task, domain, action string }
	seeds := []S{
		// Coding → coding_llm
		{"write a function", "coding", "coding_llm"},
		{"debug this code", "coding", "coding_llm"},
		{"fix the bug", "coding", "coding_llm"},
		{"implement feature", "coding", "coding_llm"},
		{"code review", "coding", "coding_llm"},
		{"unit tests", "coding", "coding_llm"},
		{"refactor", "coding", "coding_llm"},
		{"compile error", "coding", "coding_llm"},
		{"python script", "coding", "coding_llm"},
		{"javascript", "coding", "coding_llm"},
		{"SQL query", "coding", "coding_llm"},
		{"REST API", "coding", "coding_llm"},
		{"algorithm", "coding", "coding_llm"},
		{"data structure", "coding", "coding_llm"},
		// Content → content_llm
		{"write a blog post", "content", "content_llm"},
		{"create article", "content", "content_llm"},
		{"SEO copy", "content", "content_llm"},
		{"write an email", "content", "content_llm"},
		{"social media post", "content", "content_llm"},
		{"marketing copy", "content", "content_llm"},
		{"newsletter", "content", "content_llm"},
		{"product description", "content", "content_llm"},
		{"press release", "content", "content_llm"},
		// Media → media_llm
		{"generate image", "media", "media_llm"},
		{"create video", "media", "media_llm"},
		{"design graphic", "media", "media_llm"},
		{"image for", "media", "media_llm"},
		{"illustration", "media", "media_llm"},
		// Finance → nn_finance
		{"investment strategy", "finance", "nn_finance"},
		{"stock analysis", "finance", "nn_finance"},
		{"financial report", "finance", "nn_finance"},
		{"budget planning", "finance", "nn_finance"},
		{"ROI calculation", "finance", "nn_finance"},
		{"portfolio", "finance", "nn_finance"},
		{"revenue forecast", "finance", "nn_finance"},
		{"tax planning", "finance", "nn_finance"},
		{"valuation", "finance", "nn_finance"},
		{"cash flow", "finance", "nn_finance"},
		// Legal → nn_legal
		{"legal advice", "legal", "nn_legal"},
		{"contract review", "legal", "nn_legal"},
		{"compliance", "legal", "nn_legal"},
		{"terms of service", "legal", "nn_legal"},
		{"privacy policy", "legal", "nn_legal"},
		{"intellectual property", "legal", "nn_legal"},
		{"GDPR", "legal", "nn_legal"},
		{"copyright", "legal", "nn_legal"},
		{"lawsuit", "legal", "nn_legal"},
		// Medical → nn_medical
		{"medical question", "medical", "nn_medical"},
		{"symptoms", "medical", "nn_medical"},
		{"treatment options", "medical", "nn_medical"},
		{"drug interaction", "medical", "nn_medical"},
		{"health advice", "medical", "nn_medical"},
		{"diagnosis", "medical", "nn_medical"},
		// Data → nn_data
		{"data analysis", "data", "nn_data"},
		{"statistical model", "data", "nn_data"},
		{"chart this data", "data", "nn_data"},
		{"regression", "data", "nn_data"},
		{"machine learning model", "data", "nn_data"},
		{"dataset", "data", "nn_data"},
		{"visualise data", "data", "nn_data"},
		// Science → nn_science
		{"scientific research", "science", "nn_science"},
		{"explain physics", "science", "nn_science"},
		{"chemistry", "science", "nn_science"},
		{"biology", "science", "nn_science"},
		{"research methodology", "science", "nn_science"},
		// System tools
		{"search the web", "search", "web_search"},
		{"find information", "search", "web_search"},
		{"look up", "search", "web_search"},
		{"fetch url", "search", "fetch_url"},
		{"run command", "system", "run_command"},
		{"execute shell", "system", "run_command"},
		{"write file", "system", "write_file"},
		{"read file", "system", "read_file"},
		{"remember this", "memory", "store_memory"},
		{"recall what", "memory", "recall_memory"},
		{"create task", "tasks", "create_task"},
		{"list my tasks", "tasks", "list_tasks"},
		{"send discord", "comms", "send_discord"},
		{"send message", "comms", "send_telegram"},
	}
	for _, s := range seeds {
		nn1.store(s.task, s.domain+":"+s.action)
	}
	auditLog("NEURAL", fmt.Sprintf("NN1 seeded with %d routing patterns", len(seeds)))
}

// ── toolDomain: map tool name → domain ───────────────────────────────────────

func toolDomain(toolName string) string {
	switch {
	case toolName == "coding_llm" || strings.Contains(toolName, "code"):
		return "coding"
	case toolName == "content_llm":
		return "content"
	case toolName == "media_llm":
		return "media"
	case toolName == "web_search" || toolName == "fetch_url":
		return "search"
	case toolName == "run_command" || toolName == "write_file" || toolName == "read_file":
		return "system"
	case strings.HasPrefix(toolName, "store_") || strings.HasPrefix(toolName, "recall_"):
		return "memory"
	case strings.HasPrefix(toolName, "create_task") || toolName == "list_tasks" ||
		toolName == "complete_task" || toolName == "update_task":
		return "tasks"
	case strings.HasPrefix(toolName, "send_"):
		return "comms"
	case strings.HasPrefix(toolName, "nn_"):
		return "neural"
	}
	return ""
}

// learnFromLiveCall is called after every successful tool execution.
func learnFromLiveCall(taskDescription, toolName string) {
	domain := toolDomain(toolName)
	if domain == "" {
		return
	}
	// NN1 always learns the routing pattern
	chain.learnRoute(taskDescription, domain, toolName)
	// Expert NN learns ONLY if it is already awake (avoid forcing a disk load for live training)
	chain.mu.RLock()
	awakeEx, isAwake := chain.awake[domain]
	chain.mu.RUnlock()
	if isAwake && awakeEx != nil {
		awakeEx.store(taskDescription, "tool:"+toolName)
		chain.resetSleepTimer(domain) // extend wake window since we just taught it
	}
	auditLog("NEURAL", fmt.Sprintf("Live route learned: [%.40s] → %s/%s (expert_awake=%v)",
		taskDescription, domain, toolName, isAwake))
}

// ── Expert context builder (called before every task) ────────────────────────

// buildExpertChainContext is called before every task.
// NN1 classifies the task, then ONLY IF confidence is high enough does it
// wake the relevant expert NN from disk. The expert is put back to sleep
// automatically after expertSleepAfter seconds via the auto-sleep timer.
func buildExpertChainContext(task string) string {
	if chain.nn1 == nil {
		return ""
	}
	domain, action, conf := chain.classify(task)
	var sections []string

	// NN1 routing summary (always included if classified)
	if domain != "" && conf > 0.30 {
		sections = append(sections, fmt.Sprintf(
			"── NN1 Routing ──\ndomain=%s  action=%s  confidence=%.2f", domain, action, conf))
	}

	// Wake expert ONLY when NN1 is confident enough — preserves dormancy otherwise
	if domain != "" && conf > 0.45 {
		if ex, ok := chain.wakeExpert(domain); ok {
			// Expert is now awake with W matrices loaded — query it
			results := ex.recall(task, 5)
			// Expert stays awake until auto-sleep timer fires (90s)
			// sleepExpert is NOT called here — timer handles it
			var lines []string
			for _, r := range results {
				if r.Score < 0.12 {
					break
				}
				lines = append(lines, fmt.Sprintf("[%.3f] %s → %s", r.Score, r.Key, r.Value))
			}
			if len(lines) > 0 {
				sections = append(sections, fmt.Sprintf("── Expert [%s] (scale=%d, trained=%d) ──\n%s",
					strings.ToUpper(domain), ex.ScaleLevel, ex.TrainCount, strings.Join(lines, "\n")))
			}
		}
	}

	// High confidence routing hint
	if conf > 0.85 {
		if stub, ok := chain.getStub(domain); ok && stub.TrainCount > 15 {
			sections = append(sections, fmt.Sprintf(
				"[NN1: high confidence — delegate directly to %s]", action))
		}
	}

	if len(sections) == 0 {
		return ""
	}
	return strings.Join(sections, "\n\n")
}

// ── Self-Training Delegator ───────────────────────────────────────────────────

func expertTrainingPrompts(domain string, skills []string) []string {
	base := []string{
		fmt.Sprintf("Give one key %s concept as JSON {\"key\":\"concept name\",\"value\":\"explanation\"}", domain),
		fmt.Sprintf("Give a %s best practice as JSON {\"key\":\"practice\",\"value\":\"how to apply it\"}", domain),
		fmt.Sprintf("Give a common %s mistake and fix as JSON {\"key\":\"mistake\",\"value\":\"correct approach\"}", domain),
		fmt.Sprintf("Give a %s workflow step as JSON {\"key\":\"step name\",\"value\":\"what to do\"}", domain),
		fmt.Sprintf("Give a %s tool or framework as JSON {\"key\":\"tool name\",\"value\":\"what it does\"}", domain),
		fmt.Sprintf("Give a %s formula or rule as JSON {\"key\":\"rule name\",\"value\":\"formula or rule\"}", domain),
	}
	for _, s := range skills {
		base = append(base, fmt.Sprintf("Explain %s in %s as JSON {\"key\":\"%s\",\"value\":\"explanation\"}", s, domain, s))
	}
	return base
}

func expertStaticKnowledge(domain string) [][2]string {
	static := map[string][][2]string{
		"finance": {
			{"ROI formula", "ROI = (Net Profit / Cost) × 100"},
			{"compound interest", "A = P(1 + r/n)^(nt)"},
			{"P/E ratio", "Price per Share / Earnings per Share — higher = more expensive"},
			{"liquidity ratio", "Current Ratio = Current Assets / Current Liabilities, healthy >1.5"},
			{"DCF valuation", "NPV = Σ CF_t/(1+r)^t — discount future cash flows"},
			{"diversification", "Spread across asset classes to reduce unsystematic risk"},
			{"Sharpe ratio", "Sharpe = (Return − RiskFreeRate) / StdDev — reward per unit risk"},
			{"CAPM", "E(R) = Rf + β(Rm−Rf) — expected return from market risk"},
		},
		"legal": {
			{"contract elements", "Offer + acceptance + consideration + capacity + legality"},
			{"GDPR lawful basis", "Consent, contract, legal obligation, vital interests, public task, legitimate interests"},
			{"IP types", "Copyright=expression, Patent=invention, Trademark=brand, Trade Secret=confidential"},
			{"negligence test", "Duty of care + breach + causation + damages"},
			{"force majeure", "Clause excusing performance due to extraordinary unforeseeable events"},
			{"indemnification", "One party agrees to compensate the other for specified losses"},
			{"jurisdiction clause", "Specifies which court/law governs disputes"},
		},
		"medical": {
			{"normal vitals", "BP 120/80, HR 60-100bpm, Temp 36.5-37.5°C, SpO2 95-100%"},
			{"BMI categories", "Underweight<18.5, Normal 18.5-24.9, Overweight 25-29.9, Obese≥30"},
			{"triage levels", "Immediate(red), Delayed(yellow), Minimal(green), Expectant(black)"},
			{"CYP450 note", "Most drugs metabolised by CYP450 — inducers/inhibitors affect blood levels"},
			{"pain scale", "0=none, 1-3=mild, 4-6=moderate, 7-9=severe, 10=worst imaginable"},
		},
		"data": {
			{"overfitting fixes", "Regularisation L1/L2, dropout, early stopping, cross-validation, more data"},
			{"correlation note", "Correlation −1 to +1 measures linear relationship, not causation"},
			{"bias-variance", "High bias=underfitting, high variance=overfitting — tune via CV"},
			{"data cleaning", "Handle nulls, deduplicate, normalise, encode categoricals, remove outliers"},
			{"train-test split", "Typically 80/20 or 70/30 — never fit on test data"},
			{"confusion matrix", "TP, TN, FP, FN — derive precision, recall, F1"},
		},
		"science": {
			{"scientific method", "Observe → Hypothesise → Predict → Experiment → Analyse → Conclude"},
			{"Newton laws", "1=inertia, 2=F=ma, 3=action-reaction"},
			{"thermodynamics", "0=equilibrium, 1=energy conserved, 2=entropy increases, 3=abs zero unattainable"},
			{"evolution", "Natural selection: variation + heredity + selection pressure → adaptation"},
		},
		"coding": {
			{"SOLID", "Single Responsibility, Open/Closed, Liskov, Interface Segregation, Dependency Inversion"},
			{"Big O", "O(1)=hash O(logN)=binary O(N)=linear O(NlogN)=sort O(N²)=nested"},
			{"REST verbs", "GET=read POST=create PUT=replace PATCH=update DELETE=remove"},
			{"git flow", "feature-branch → commit → PR → review → merge → delete branch"},
			{"ACID", "Atomicity Consistency Isolation Durability — DB transaction properties"},
			{"CAP theorem", "Can only guarantee 2 of: Consistency, Availability, Partition tolerance"},
		},
		"content": {
			{"headline formula", "Number + Adjective + Keyword + Promise"},
			{"SEO basics", "Target keyword in title, H1, first 100 words, meta desc, alt text"},
			{"content funnel", "TOFU=awareness(blog), MOFU=consideration(guides), BOFU=decision(demos)"},
			{"readability", "Short sentences, active voice, subheadings every 300 words, bullets for lists"},
			{"CTA formula", "Verb + benefit + urgency. E.g. 'Get free access — limited time'"},
		},
	}
	if pairs, ok := static[domain]; ok {
		return pairs
	}
	return [][2]string{}
}

func trainExpertFromLLM(ex *ExpertNN) {
	trainingMu.Lock()
	job := &TrainingJob{ExpertID: ex.ID, ExpertName: ex.Name, Mode: "llm",
		Status: "running", StartedAt: time.Now()}
	trainingJobs = append(trainingJobs, job)
	trainingMu.Unlock()

	ex.Status = "training"
	auditLog("NEURAL", fmt.Sprintf("LLM-training Expert '%s'...", ex.Domain))

	mu.Lock()
	url, model, key := mainLLMURL, mainLLMModel, mainLLMKey
	mu.Unlock()
	if url == "" {
		job.Status = "error"
		job.Message = "Main LLM not configured"
		ex.Status = "sleeping"
		return
	}

	prompts := expertTrainingPrompts(ex.Domain, ex.SkillSet)
	job.Total = len(prompts)

	for i, prompt := range prompts {
		msgs := []Message{
			{Role: "system", Content: fmt.Sprintf(
				"You are a %s expert. Respond ONLY with valid JSON: {\"key\":\"short concept\",\"value\":\"concise explanation\"}. No preamble, no markdown.", ex.Domain)},
			{Role: "user", Content: prompt},
		}
		resp, err := callLLM(url, model, key, msgs)
		if err != nil {
			job.Progress = i + 1
			continue
		}
		resp = strings.TrimSpace(resp)
		for _, pfx := range []string{"```json", "```"} {
			resp = strings.TrimPrefix(resp, pfx)
		}
		resp = strings.TrimSuffix(resp, "```")
		var pair map[string]string
		if json.Unmarshal([]byte(strings.TrimSpace(resp)), &pair) == nil {
			if k, v := pair["key"], pair["value"]; k != "" && v != "" {
				ex.store(k, v)
				chain.learnRoute(k, ex.Domain, "nn_"+ex.Domain)
			}
		}
		job.Progress = i + 1
		time.Sleep(150 * time.Millisecond)
	}

	job.Status = "done"
	job.FinishedAt = time.Now()
	ex.Status = "sleeping"
	go ex.save()
	auditLog("NEURAL", fmt.Sprintf("LLM-training done for '%s': %d pairs", ex.Domain, ex.TrainCount))
}

func trainExpertSelf(ex *ExpertNN) {
	trainingMu.Lock()
	job := &TrainingJob{ExpertID: ex.ID, ExpertName: ex.Name, Mode: "self",
		Status: "running", StartedAt: time.Now()}
	trainingJobs = append(trainingJobs, job)
	trainingMu.Unlock()

	ex.Status = "training"
	auditLog("NEURAL", fmt.Sprintf("Self-training Expert '%s'...", ex.Domain))

	// Pull domain-relevant entries from base NN key store (avoid type mismatch)
	nn, ok := nnCtrl.get(NN_DEFAULT)
	var baseEntries []NNEntry
	if ok && nn != nil {
		for _, e := range nn.Keys {
			if containsAny(strings.ToLower(e.Key+" "+e.Value), []string{ex.Domain}) {
				baseEntries = append(baseEntries, e)
			}
		}
	}
	staticPairs := expertStaticKnowledge(ex.Domain)
	job.Total = len(baseEntries) + len(staticPairs)
	idx := 0
	for _, e := range baseEntries {
		ex.store(e.Key, e.Value)
		idx++; job.Progress = idx
	}
	for _, p := range staticPairs {
		ex.store(p[0], p[1])
		chain.learnRoute(p[0], ex.Domain, "nn_"+ex.Domain)
		idx++; job.Progress = idx
	}

	job.Status = "done"
	job.FinishedAt = time.Now()
	// NOTE: caller (toolTrainExpert or toolSpawnExpert goroutine) handles sleepExpert
	auditLog("NEURAL", fmt.Sprintf("Self-training done for '%s': %d pairs stored", ex.Domain, ex.TrainCount))
}

// ── NNChain recall proxy (used by base NN controller recall) ──────────────────

func (c *NNController) recallFromExpert(domain, query string, topK int) string {
	if ex, ok := chain.get(domain); ok {
		ex.Status = "awake"
		results := ex.recall(query, topK)
		ex.Status = "sleeping"
		var lines []string
		for _, r := range results {
			if r.Score < 0.10 {
				break
			}
			lines = append(lines, fmt.Sprintf("[%.3f] %s → %s", r.Score, r.Key, r.Value))
		}
		if len(lines) > 0 {
			return strings.Join(lines, "\n")
		}
	}
	return ""
}

// ── Chain initialisation ──────────────────────────────────────────────────────

func initNNChain() {
	// ── NN1: Master Router — loaded fully into RAM, always awake ──────────────
	nn1, err := loadExpertNN("nn1_master")
	if err != nil {
		// First run: create + seed NN1 with routing knowledge
		nn1 = newExpertNN("nn1_master", "NN1 Master Router", "routing",
			[]string{"task classification", "expert delegation", "tool routing", "online learning"}, 2, "seed")
		seedNN1(nn1)
		go nn1.save()
		auditLog("NEURAL", fmt.Sprintf("NN1 created and seeded with %d routing patterns", nn1.TrainCount))
	} else {
		auditLog("NEURAL", fmt.Sprintf("NN1 loaded — %d routing patterns, %d wakes", nn1.TrainCount, nn1.WakeCount))
	}
	nn1.Status = "awake" // NN1 is ALWAYS awake — never sleeps
	chain.nn1 = nn1

	// ── Expert NNs: register stubs ONLY — no W matrices loaded ───────────────
	// W matrices are loaded on demand by wakeExpert() when NN1 routes a task.
	type def struct {
		id, name, domain string
		skills           []string
		scale            int
	}
	defaults := []def{
		{"exp_coding",  "Coding Expert",  "coding",
			[]string{"debugging", "algorithms", "code review", "system design", "testing"}, 2},
		{"exp_content", "Content Expert", "content",
			[]string{"SEO writing", "blog posts", "social media", "email copy", "marketing"}, 2},
		{"exp_media",   "Media Expert",   "media",
			[]string{"image prompts", "video concepts", "graphic design briefs"}, 1},
		{"exp_finance", "Finance Expert", "finance",
			[]string{"investment analysis", "financial modelling", "risk assessment", "valuation", "tax"}, 4},
		{"exp_legal",   "Legal Expert",   "legal",
			[]string{"contracts", "IP law", "compliance", "GDPR", "litigation"}, 4},
		{"exp_medical", "Medical Expert", "medical",
			[]string{"symptoms", "treatments", "pharmacology", "triage", "nutrition"}, 4},
		{"exp_data",    "Data Expert",    "data",
			[]string{"ML models", "statistics", "data cleaning", "visualisation", "SQL"}, 2},
		{"exp_science", "Science Expert", "science",
			[]string{"physics", "chemistry", "biology", "research methods", "mathematics"}, 2},
	}

	for _, d := range defaults {
		// Try to read metadata from disk without loading W matrices
		stub, err := loadExpertStub(d.id)
		if err != nil {
			// New expert — create, seed, save to disk, register stub (no RAM kept)
			ex := newExpertNN(d.id, d.name, d.domain, d.skills, d.scale, "seed")
			for _, p := range expertStaticKnowledge(d.domain) {
				ex.store(p[0], p[1])
				chain.learnRoute(p[0], d.domain, "nn_"+d.domain)
			}
			ex.save()        // persist W matrices to disk
			ex.Layers = nil  // immediately free W matrices from RAM
			stub = ex
			auditLog("NEURAL", fmt.Sprintf("Expert seeded+dormant: [%s] scale=%d trained=%d", d.name, d.scale, ex.TrainCount))
		} else {
			auditLog("NEURAL", fmt.Sprintf("Expert stub loaded: [%s] trained=%d wakes=%d", stub.Name, stub.TrainCount, stub.WakeCount))
		}
		chain.registerStub(stub) // registers with Layers=nil — dormant
	}

	auditLog("NEURAL", fmt.Sprintf(
		"NN Chain online: NN1 awake (%.1f KB) + %d experts dormant (0 KB RAM each)",
		float64(2*256*256*8)/1024, len(chain.stubs)))
}

// loadExpertStub reads expert metadata from disk WITHOUT loading W matrices.
// Used at startup to populate the stub registry cheaply.
func loadExpertStub(id string) (*ExpertNN, error) {
	ex, err := loadExpertNN(id)
	if err != nil {
		return nil, err
	}
	// Discard W matrices immediately — we only want metadata
	ex.Layers = nil
	ex.Status = "sleeping"
	return ex, nil
}

// ── Agent tools for Expert Chain ──────────────────────────────────────────────

func toolNNChainStatus(params map[string]interface{}) string {
	lines := []string{}
	if chain.nn1 != nil {
		chain.mu.RLock()
		routeCount := len(chain.routing)
		awakeCount := len(chain.awake)
		chain.mu.RUnlock()
		lines = append(lines, fmt.Sprintf(
			"NN1 [Master Router] AWAKE — scale=%d trained=%d routes=%d | experts_awake=%d/%d",
			chain.nn1.ScaleLevel, chain.nn1.TrainCount, routeCount, awakeCount, len(chain.stubs)))
	}
	for _, stub := range chain.list() {
		// Check if currently awake (W matrices in RAM)
		chain.mu.RLock()
		_, isAwake := chain.awake[stub.Domain]
		chain.mu.RUnlock()
		statusStr := "dormant"
		if isAwake { statusStr = "AWAKE (RAM)" }
		if stub.Status == "training" { statusStr = "training" }
		lines = append(lines, fmt.Sprintf(
			"Expert [%s] domain=%s scale=%d trained=%d wakes=%d — %s",
			stub.Name, stub.Domain, stub.ScaleLevel, stub.TrainCount, stub.WakeCount, statusStr))
	}
	return strings.Join(lines, "\n")
}

func toolSpawnExpert(params map[string]interface{}) string {
	name, _ := params["name"].(string)
	domain, _ := params["domain"].(string)
	if name == "" || domain == "" {
		return "Error: name and domain are required."
	}
	if _, ok := chain.getStub(domain); ok {
		return fmt.Sprintf("Expert for domain '%s' already exists.", domain)
	}
	skillsStr, _ := params["skills"].(string)
	var skills []string
	for _, s := range strings.Split(skillsStr, ",") {
		if t := strings.TrimSpace(s); t != "" {
			skills = append(skills, t)
		}
	}
	scaleF, _ := params["scale"].(float64)
	scale := int(scaleF)
	if scale == 0 { scale = 2 }
	id := "exp_" + strings.ToLower(strings.ReplaceAll(domain, " ", "_"))
	// Create fully hydrated expert (W matrices needed for seeding)
	ex := newExpertNN(id, name, domain, skills, scale, "nn1")
	// Register as awake so training can proceed
	chain.mu.Lock()
	chain.awake[domain] = ex
	chain.mu.Unlock()
	// Self-train in background; when done it calls registerStub and sleeps
	go func() {
		trainExpertSelf(ex)
		// After training: save to disk, register stub, free W matrices
		chain.registerStub(ex)
		chain.mu.Lock()
		delete(chain.awake, domain)
		chain.mu.Unlock()
		ex.Layers = nil // free RAM
		auditLog("NEURAL", fmt.Sprintf("Expert '%s' trained and dormant.", ex.Name))
	}()
	return fmt.Sprintf("Expert NN '%s' (domain=%s scale=%d) spawned. Self-training started; will sleep when done.", name, domain, scale)
}

func toolTrainExpert(params map[string]interface{}) string {
	domain, _ := params["domain"].(string)
	mode, _ := params["mode"].(string)
	if domain == "" {
		return "Error: domain required."
	}
	stub, ok := chain.getStub(domain)
	if !ok {
		return fmt.Sprintf("No expert for domain '%s'. Use nn_spawn_expert first.", domain)
	}
	if stub.Status == "training" {
		return fmt.Sprintf("Expert '%s' is already training.", stub.Name)
	}
	// Wake expert to get W matrices for training
	ex, ok := chain.wakeExpert(domain)
	if !ok {
		return fmt.Sprintf("Could not wake Expert '%s' for training.", stub.Name)
	}
	ex.Status = "training"
	stub.Status = "training"
	switch mode {
	case "llm":
		go func() {
			trainExpertFromLLM(ex)
			chain.sleepExpert(domain) // save + free RAM when done
		}()
		return fmt.Sprintf("LLM-training started for Expert '%s'. Will sleep when complete.", ex.Name)
	default:
		go func() {
			trainExpertSelf(ex)
			chain.sleepExpert(domain) // save + free RAM when done
		}()
		return fmt.Sprintf("Self-training started for Expert '%s'. Will sleep when complete.", ex.Name)
	}
}

func toolQueryExpert(params map[string]interface{}) string {
	domain, _ := params["domain"].(string)
	query, _ := params["query"].(string)
	if domain == "" || query == "" {
		return "Error: domain and query required."
	}
	// Wake expert on demand — loads W matrices from disk
	ex, ok := chain.wakeExpert(domain)
	if !ok {
		return fmt.Sprintf("No expert for domain '%s' (not registered or file missing).", domain)
	}
	results := ex.recall(query, 5)
	// Expert sleeps automatically after 90s idle — no explicit sleep needed here
	if len(results) == 0 || results[0].Score < 0.10 {
		return fmt.Sprintf("Expert [%s] has no relevant knowledge for that query yet.", ex.Name)
	}
	var lines []string
	for _, r := range results {
		if r.Score < 0.10 { break }
		lines = append(lines, fmt.Sprintf("[%.3f] %s → %s", r.Score, r.Key, r.Value))
	}
	return fmt.Sprintf("Expert [%s] recall:\n%s", ex.Name, strings.Join(lines, "\n"))
}

// --- TYPED MEMORY SHORTCUTS ---
func storeEpisodicMemory(params map[string]interface{}) string {
	params["tier"] = "episodic"
	if _, ok := params["importance"]; !ok {
		params["importance"] = float64(7)
	}
	return storeMemory(params)
}
func recallEpisodicMemory(params map[string]interface{}) string {
	params["tier"] = "episodic"
	return recallMemory(params)
}
func storeSemanticMemory(params map[string]interface{}) string {
	params["tier"] = "semantic"
	if _, ok := params["importance"]; !ok {
		params["importance"] = float64(8)
	}
	return storeMemory(params)
}
func recallSemanticMemory(params map[string]interface{}) string {
	params["tier"] = "semantic"
	return recallMemory(params)
}
func storeProceduralMemory(params map[string]interface{}) string {
	params["tier"] = "procedural"
	if _, ok := params["importance"]; !ok {
		params["importance"] = float64(9)
	}
	return storeMemory(params)
}
func recallProceduralMemory(params map[string]interface{}) string {
	params["tier"] = "procedural"
	return recallMemory(params)
}

// --- TASK MANAGEMENT ---
func loadTasks() {
	mu.Lock()
	defer mu.Unlock()
	data, err := os.ReadFile(TASKS_FILE)
	if err == nil {
		json.Unmarshal(data, &tasksDb)
	}
}

func saveTasksLocked() {
	data, _ := json.MarshalIndent(tasksDb, "", "  ")
	os.WriteFile(TASKS_FILE, data, 0644)
}

func createTask(params map[string]interface{}) string {
	title, _ := params["title"].(string)
	desc, _ := params["description"].(string)
	if title == "" {
		return "Error: 'title' is required."
	}
	priority := 2
	if p, ok := params["priority"].(float64); ok {
		priority = int(p)
	}
	mu.Lock()
	task := Task{
		ID:          len(tasksDb) + 1,
		Title:       title,
		Description: desc,
		Status:      "pending",
		Priority:    priority,
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
	}
	tasksDb = append(tasksDb, task)
	saveTasksLocked()
	mu.Unlock()
	auditLog("TASK", fmt.Sprintf("Created task #%d: %s", task.ID, title))
	return fmt.Sprintf("Task #%d created: %s", task.ID, title)
}

func listTasks(params map[string]interface{}) string {
	statusFilter, _ := params["status"].(string)
	if statusFilter == "" {
		statusFilter = "all"
	}
	mu.Lock()
	defer mu.Unlock()
	var lines []string
	for _, t := range tasksDb {
		if statusFilter != "all" && t.Status != statusFilter {
			continue
		}
		line := fmt.Sprintf("[#%d][%s][P%d] %s", t.ID, t.Status, t.Priority, t.Title)
		if t.Description != "" {
			line += ": " + t.Description
		}
		if t.Result != "" {
			line += " => " + t.Result
		}
		lines = append(lines, line)
	}
	if len(lines) == 0 {
		return fmt.Sprintf("No tasks with status '%s'.", statusFilter)
	}
	return strings.Join(lines, "\n")
}

func updateTask(params map[string]interface{}) string {
	idF, _ := params["id"].(float64)
	id := int(idF)
	status, _ := params["status"].(string)
	result, _ := params["result"].(string)
	if id == 0 {
		return "Error: 'id' is required."
	}
	mu.Lock()
	defer mu.Unlock()
	for i, t := range tasksDb {
		if t.ID == id {
			if status != "" {
				tasksDb[i].Status = status
			}
			if result != "" {
				tasksDb[i].Result = result
			}
			tasksDb[i].UpdatedAt = time.Now()
			saveTasksLocked()
			auditLog("TASK", fmt.Sprintf("Updated task #%d: status=%s", id, tasksDb[i].Status))
			return fmt.Sprintf("Task #%d updated.", id)
		}
	}
	return fmt.Sprintf("Task #%d not found.", id)
}

func completeTask(params map[string]interface{}) string {
	params["status"] = "complete"
	return updateTask(params)
}

// --- GO MODULE MANAGEMENT ---
func addModule(params map[string]interface{}) string {
	modPath, _ := params["module_path"].(string)
	version, _ := params["version"].(string)
	if modPath == "" {
		return "Error: 'module_path' is required."
	}
	if version == "" {
		version = "latest"
	}
	spec := modPath + "@" + version
	auditLog("MODULE", fmt.Sprintf("Adding module: %s", spec))
	cmd := exec.Command("go", "get", spec)
	cmd.Dir = goModWorkDir()
	out, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Sprintf("go get failed: %v\n%s", err, string(out))
	}
	exec.Command("go", "mod", "tidy").Run()
	auditLog("MODULE", fmt.Sprintf("Module added: %s", spec))
	if modData, merr := os.ReadFile("go.mod"); merr == nil {
		timestamp := time.Now().Format("20060102-150405")
		os.WriteFile(docPath("modules", fmt.Sprintf("gomod_%s.txt", timestamp)), modData, 0644)
	}
	return fmt.Sprintf("Added %s\n%s", spec, strings.TrimSpace(string(out)))
}

func removeModule(params map[string]interface{}) string {
	modPath, _ := params["module_path"].(string)
	if modPath == "" {
		return "Error: 'module_path' is required."
	}
	auditLog("MODULE", fmt.Sprintf("Removing module: %s", modPath))
	cmd := exec.Command("go", "get", modPath+"@none")
	cmd.Dir = goModWorkDir()
	out, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Sprintf("go get @none failed: %v\n%s", err, string(out))
	}
	exec.Command("go", "mod", "tidy").Run()
	auditLog("MODULE", fmt.Sprintf("Module removed: %s", modPath))
	return fmt.Sprintf("Removed %s\n%s", modPath, strings.TrimSpace(string(out)))
}

func listModules(params map[string]interface{}) string {
	cmd := exec.Command("go", "list", "-m", "all")
	cmd.Dir = goModWorkDir()
	out, err := cmd.CombinedOutput()
	if err != nil {
		data, ferr := os.ReadFile("go.mod")
		if ferr != nil {
			return fmt.Sprintf("Error listing modules: %v", err)
		}
		return string(data)
	}
	return strings.TrimSpace(string(out))
}

func goModWorkDir() string {
	if _, err := os.Stat("go.mod"); err == nil {
		return "."
	}
	dir := strings.TrimSuffix(SOURCE_FILE, "agent.go")
	if dir == "" {
		dir = "."
	}
	return dir
}

func readGoMod() string {
	data, err := os.ReadFile("go.mod")
	if err != nil {
		return ""
	}
	return string(data)
}

func editOwnCode(params map[string]interface{}) string {
	search, _ := params["search_string"].(string)
	replacement, _ := params["replacement_string"].(string)
	auditLog("MODIFICATION", "Editing source code...")
	content, err := os.ReadFile(SOURCE_FILE)
	if err != nil {
		return "Error reading self."
	}
	if !strings.Contains(string(content), search) {
		return "Search string not found in code."
	}
	newContent := strings.Replace(string(content), search, replacement, 1)
	os.WriteFile(SOURCE_FILE+".bak", content, 0644)
	os.WriteFile(SOURCE_FILE, []byte(newContent), 0644)
	auditLog("SUCCESS", "Code updated.")
	return "Code updated successfully."
}

// --- SHARED LLM CALLER ---
func callLLM(urlBase, model, key string, messages []Message) (string, error) {
	if urlBase == "" {
		return "", fmt.Errorf("LLM URL not configured")
	}
	if model == "" {
		return "", fmt.Errorf("LLM model not configured")
	}
	endpoint := strings.TrimRight(urlBase, "/")
	if !strings.HasSuffix(endpoint, "/chat/completions") {
		endpoint += "/v1/chat/completions"
	}
	reqBody := OpenAIRequest{Model: model, Messages: messages}
	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return "", err
	}
	mu.Lock()
	timeoutSec := llmTimeoutSec
	mu.Unlock()
	if timeoutSec <= 0 {
		timeoutSec = 300
	}
	client := &http.Client{Timeout: time.Duration(timeoutSec) * time.Second}
	const maxRetries = 2
	var lastErr error
	for attempt := 0; attempt <= maxRetries; attempt++ {
		if attempt > 0 {
			auditLog("SYSTEM", fmt.Sprintf("LLM retry %d/%d for %s", attempt, maxRetries, model))
			time.Sleep(time.Duration(attempt*2) * time.Second)
		}
		httpReq, err := http.NewRequest("POST", endpoint, bytes.NewBuffer(jsonData))
		if err != nil {
			return "", err
		}
		httpReq.Header.Set("Content-Type", "application/json")
		if key != "" {
			httpReq.Header.Set("Authorization", "Bearer "+key)
		}
		resp, err := client.Do(httpReq)
		if err != nil {
			errStr := err.Error()
			if strings.Contains(errStr, "context deadline exceeded") || strings.Contains(errStr, "Client.Timeout") {
				lastErr = fmt.Errorf("timeout after %ds — model '%s' at %s is not responding. "+
					"Try increasing the timeout in Main LLM settings or check that Ollama/your server is running.", timeoutSec, model, urlBase)
			} else if strings.Contains(errStr, "connection refused") {
				lastErr = fmt.Errorf("connection refused — is Ollama/your LLM server running at %s?", urlBase)
			} else if strings.Contains(errStr, "no such host") {
				lastErr = fmt.Errorf("host not found: %s — check the Base URL in Main LLM settings", urlBase)
			} else {
				lastErr = fmt.Errorf("request failed: %v", err)
			}
			continue
		}
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		if resp.StatusCode == 503 || resp.StatusCode == 502 {
			lastErr = fmt.Errorf("server unavailable (HTTP %d) — model may still be loading", resp.StatusCode)
			continue
		}
		if resp.StatusCode == 404 {
			return "", fmt.Errorf("model '%s' not found (HTTP 404) — run: ollama pull %s", model, model)
		}
		if resp.StatusCode != 200 {
			return "", fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(body))
		}
		var openAIResp OpenAIResponse
		if err := json.Unmarshal(body, &openAIResp); err != nil {
			return "", fmt.Errorf("could not parse response: %v", err)
		}
		if len(openAIResp.Choices) == 0 {
			return "", fmt.Errorf("LLM returned no choices")
		}
		return openAIResp.Choices[0].Message.Content, nil
	}
	return "", lastErr
}

// --- MAIN LLM ---
func chatWithMainLLM(messages []Message) (string, error) {
	mu.Lock()
	url, model, key := mainLLMURL, mainLLMModel, mainLLMKey
	mu.Unlock()
	return callLLM(url, model, key, messages)
}

// --- CODING LLM TOOL ---
func chatWithCodingLLM(prompt string) (string, error) {
	mu.Lock()
	url, model, key := codingLLMURL, codingLLMModel, codingLLMKey
	personality := currentPersonality
	mu.Unlock()
	if url == "" {
		return "", fmt.Errorf("coding LLM URL not configured — set it in the Coding LLM tab")
	}
	sysPrompt := "You are an expert software engineer. Provide clear, concise, correct code and explanations."
	if personality != "" {
		sysPrompt += "\n\nPERSONALITY & BEHAVIOR TRAITS (follow these strictly):\n" + personality
	}
	messages := []Message{
		{Role: "system", Content: sysPrompt},
		{Role: "user", Content: prompt},
	}
	return callLLM(url, model, key, messages)
}

func codingLLMTool(params map[string]interface{}) string {
	prompt, _ := params["prompt"].(string)
	if prompt == "" {
		return "Error: 'prompt' argument is required."
	}
	mu.Lock()
	llmURL, llmModel, enabled := codingLLMURL, codingLLMModel, delegateCoding
	mu.Unlock()
	if !enabled {
		return "[Coding LLM delegation disabled] Please handle this coding task directly using your own capabilities."
	}
	if llmURL == "" {
		return "Coding LLM not configured. Handle this coding task directly."
	}
	auditLog("CODING_LLM", fmt.Sprintf("Delegating to coding LLM (%s @ %s): %s", llmModel, llmURL, truncate(prompt, 80)))
	result, err := chatWithCodingLLM(prompt)
	if err != nil {
		auditLog("ERROR", fmt.Sprintf("Coding LLM error: %v", err))
		return fmt.Sprintf("Coding LLM Error: %v", err)
	}
	auditLog("CODING_LLM", fmt.Sprintf("Coding LLM responded (%d chars)", len(result)))
	timestamp := time.Now().Format("20060102-150405")
	outFile := docPath("code", fmt.Sprintf("coding_%s.md", timestamp))
	os.WriteFile(outFile, []byte(fmt.Sprintf("# Coding Output\n**Prompt:** %s\n\n%s", truncate(prompt, 200), result)), 0644)
	auditLog("CODING_LLM", fmt.Sprintf("Saved to %s", outFile))
	return result
}

// --- MEDIA LLM TOOL ---
func chatWithMediaLLM(prompt string) (string, error) {
	mu.Lock()
	url, model, key := mediaLLMURL, mediaLLMModel, mediaLLMKey
	personality := currentPersonality
	mu.Unlock()
	if url == "" {
		return "", fmt.Errorf("media LLM URL not configured — set it in the Media LLM tab")
	}
	sysPrompt := "You are an expert AI specializing in image and video generation, editing, and media creation. Provide creative, high-quality media outputs."
	if personality != "" {
		sysPrompt += "\n\nPERSONALITY & BEHAVIOR TRAITS (follow these strictly):\n" + personality
	}
	messages := []Message{
		{Role: "system", Content: sysPrompt},
		{Role: "user", Content: prompt},
	}
	return callLLM(url, model, key, messages)
}

func mediaLLMTool(params map[string]interface{}) string {
	prompt, _ := params["prompt"].(string)
	if prompt == "" {
		return "Error: 'prompt' argument is required."
	}
	mu.Lock()
	llmURL, llmModel, enabled := mediaLLMURL, mediaLLMModel, delegateMedia
	mu.Unlock()
	if !enabled {
		return "[Media LLM delegation disabled] Describe the image/video you would generate, or indicate it cannot be produced without a media model."
	}
	if llmURL == "" {
		return "Media LLM not configured. Describe what the image/video would contain."
	}
	auditLog("MEDIA_LLM", fmt.Sprintf("Delegating to media LLM (%s @ %s): %s", llmModel, llmURL, truncate(prompt, 80)))
	result, err := chatWithMediaLLM(prompt)
	if err != nil {
		auditLog("ERROR", fmt.Sprintf("Media LLM error: %v", err))
		return fmt.Sprintf("Media LLM Error: %v", err)
	}
	auditLog("MEDIA_LLM", fmt.Sprintf("Media LLM responded (%d chars)", len(result)))
	timestamp := time.Now().Format("20060102-150405")
	outFile := docPath("media", fmt.Sprintf("media_%s.md", timestamp))
	os.WriteFile(outFile, []byte(fmt.Sprintf("# Media Output\n**Prompt:** %s\n\n%s", truncate(prompt, 200), result)), 0644)
	auditLog("MEDIA_LLM", fmt.Sprintf("Saved to %s", outFile))
	return result
}

// --- CONTENT LLM TOOL ---
func chatWithContentLLM(prompt string) (string, error) {
	mu.Lock()
	url, model, key := contentLLMURL, contentLLMModel, contentLLMKey
	personality := currentPersonality
	mu.Unlock()
	if url == "" {
		return "", fmt.Errorf("content LLM URL not configured — set it in the Content LLM tab")
	}
	sysPrompt := "You are an expert web content writer and digital marketer. Produce high-quality blog posts, articles, social copy, SEO content, and web writing."
	if personality != "" {
		sysPrompt += "\n\nPERSONALITY & BEHAVIOR TRAITS (follow these strictly):\n" + personality
	}
	messages := []Message{
		{Role: "system", Content: sysPrompt},
		{Role: "user", Content: prompt},
	}
	return callLLM(url, model, key, messages)
}

func contentLLMTool(params map[string]interface{}) string {
	prompt, _ := params["prompt"].(string)
	if prompt == "" {
		return "Error: 'prompt' argument is required."
	}
	mu.Lock()
	llmURL, llmModel, enabled := contentLLMURL, contentLLMModel, delegateContent
	mu.Unlock()
	if !enabled {
		return "[Content LLM delegation disabled] Write this web content directly using your own capabilities."
	}
	if llmURL == "" {
		return "Content LLM not configured. Write this content directly."
	}
	auditLog("CONTENT_LLM", fmt.Sprintf("Delegating to content LLM (%s @ %s): %s", llmModel, llmURL, truncate(prompt, 80)))
	result, err := chatWithContentLLM(prompt)
	if err != nil {
		auditLog("ERROR", fmt.Sprintf("Content LLM error: %v", err))
		return fmt.Sprintf("Content LLM Error: %v", err)
	}
	auditLog("CONTENT_LLM", fmt.Sprintf("Content LLM responded (%d chars)", len(result)))
	timestamp := time.Now().Format("20060102-150405")
	outFile := docPath("content", fmt.Sprintf("content_%s.md", timestamp))
	os.WriteFile(outFile, []byte(fmt.Sprintf("# Content Output\n**Prompt:** %s\n\n%s", truncate(prompt, 200), result)), 0644)
	auditLog("CONTENT_LLM", fmt.Sprintf("Saved to %s", outFile))
	return result
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

var tools = map[string]func(map[string]interface{}) string{
	"write_file":               writeFile,
	"read_file":                readFile,
	"list_directory":           func(p map[string]interface{}) string { return "Use run_command 'ls -la'" },
	"run_command":              runCommand,
	"store_memory":             storeMemory,
	"recall_memory":            recallMemory,
	"store_episodic_memory":    storeEpisodicMemory,
	"recall_episodic_memory":   recallEpisodicMemory,
	"store_semantic_memory":    storeSemanticMemory,
	"recall_semantic_memory":   recallSemanticMemory,
	"store_procedural_memory":  storeProceduralMemory,
	"recall_procedural_memory": recallProceduralMemory,
	"edit_own_code":            editOwnCode,
	"coding_llm":               codingLLMTool,
	"media_llm":                mediaLLMTool,
	"content_llm":              contentLLMTool,
	"create_task":              createTask,
	"list_tasks":               listTasks,
	"update_task":              updateTask,
	"complete_task":            completeTask,
	"web_search":               webSearch,
	"fetch_url":                fetchURL,
	"send_discord":             sendDiscord,
	"send_telegram":            sendTelegram,
	"send_slack":               sendSlack,
	"send_twitter":             sendTwitter,
	"send_whatsapp":            sendWhatsApp,
	"add_module":               addModule,
	"remove_module":            removeModule,
	"list_modules":             func(p map[string]interface{}) string { return listModules(p) },
	"nn_spawn":                  toolNNSpawn,
	"nn_store":                  toolNNStore,
	"nn_recall":                 toolNNRecall,
	"nn_forget":                 toolNNForget,
	"nn_list":                   toolNNList,
	"nn_stats":                  toolNNStats,
	"nn_chain_status":           toolNNChainStatus,
	"nn_spawn_expert":           toolSpawnExpert,
	"nn_train_expert":           toolTrainExpert,
	"nn_query_expert":           toolQueryExpert,
}

// --- CHAT SESSION MANAGEMENT ---
func loadChatSessions() {
	mu.Lock()
	defer mu.Unlock()
	data, err := os.ReadFile(SESSIONS_FILE)
	if err == nil {
		json.Unmarshal(data, &chatSessions)
	}
}

func saveChatSessionsLocked() {
	data, _ := json.MarshalIndent(chatSessions, "", "  ")
	os.WriteFile(SESSIONS_FILE, data, 0644)
}

func newChat() ChatSession {
	mu.Lock()
	defer mu.Unlock()
	sessionLogs := make([]AuditEntry, len(auditLogs))
	copy(sessionLogs, auditLogs)
	title := "Untitled Session"
	for _, entry := range sessionLogs {
		if entry.Level == "INFO" {
			title = entry.Content
			if len(title) > 60 {
				title = title[:60] + "..."
			}
			break
		}
	}
	session := ChatSession{
		ID:        len(chatSessions) + 1,
		Title:     title,
		CreatedAt: time.Now(),
		Logs:      sessionLogs,
	}
	chatSessions = append(chatSessions, session)
	saveChatSessionsLocked()
	auditLogs = []AuditEntry{}
	chatMessages = []ChatMessage{}
	return session
}

// --- HEARTBEAT SYSTEM ---
func startHeartbeat() {
	go func() {
		for {
			time.Sleep(6 * time.Hour)
			consolidateMemory()
		}
	}()
	auditLog("SYSTEM", fmt.Sprintf("Heartbeat started. Interval: %v", HEARTBEAT_INTERVAL))
	ticker := time.NewTicker(HEARTBEAT_INTERVAL)
	for range ticker.C {
		auditLog("HEARTBEAT", "Waking up to check for pending tasks...")
		mu.Lock()
		pendingCount := 0
		for _, t := range tasksDb {
			if t.Status == "pending" {
				pendingCount++
			}
		}
		mu.Unlock()
		heartbeatMsg := fmt.Sprintf("AUTONOMOUS MAINTENANCE: Use list_tasks(status=pending) to find pending work and execute it. There are currently %d pending tasks. Review audit logs for any errors.", pendingCount)
		// Heartbeat ONLY feeds NN1 via taskQueue.
		// NN1 is the only NN that may wake from heartbeat activity.
		// Expert NNs are woken by NN1's routing logic ONLY — never directly.
		taskQueue <- heartbeatMsg
	}
}

// --- AGENT LOOP ---
func runAgentLoop() {
	nnInitDir()
	loadMemory()
	loadTasks()
	loadChatSessions()
	nnCtrl.loadAll()
	mu.Lock()
	selfAgents = []*SelfAgent{}
	mu.Unlock()

	nnCtrl.spawn(NN_DEFAULT, 5000)
	nnCtrl.spawn(NN_IDENTITY, 1000)
	nnCtrl.spawn(NN_SEMANTIC, 3000)
	nnCtrl.spawn(NN_PROCEDURAL, 2000)
	nnCtrl.spawn(NN_CODE, 2000)
	auditLog("NEURAL", "All NN specialist networks online — agent brain initialised")
	initNNChain() // NN1 master router + expert specialist NNs
	for task := range taskQueue {
		auditLog("INFO", fmt.Sprintf("Task: %s", task))
		relevantContext := findRelevantMemories(task)
		messages := []Message{{Role: "system", Content: buildSystemPrompt()}}
		nnContext := nnBuildAgentContext(task)
		expertContext := buildExpertChainContext(task)
		if nnContext != "" || expertContext != "" {
			header := "╔═ NEURAL BRAIN CONTEXT ═══════════════════════════════════════╗\n" +
				"║ This is your persistent memory — valid regardless of which  ║\n" +
				"║ LLM model is currently active. It IS your brain.            ║\n" +
				"╚═══════════════════════════════════════════════════════════════╝\n"
			messages = append(messages, Message{
				Role:    "system",
				Content: header + nnContext,
			})
			auditLog("NEURAL", fmt.Sprintf("Brain context injected (%d chars) for: %.50s", len(nnContext), task))
		}
		if len(relevantContext) > 0 {
			messages = append(messages, Message{
				Role:    "system",
				Content: fmt.Sprintf("RELEVANT MEMORY CONTEXT:\n%s", strings.Join(relevantContext, "\n")),
			})
			auditLog("CONTEXT", fmt.Sprintf("Injected %d memories.", len(relevantContext)))
		}
		taskMsg := task
		lc := strings.ToLower(strings.TrimSpace(task))
		isConversational := len(strings.Fields(lc)) <= 15 &&
			!strings.Contains(lc, "write ") && !strings.Contains(lc, "create ") &&
			!strings.Contains(lc, "build ") && !strings.Contains(lc, "run ") &&
			!strings.Contains(lc, "install ") && !strings.Contains(lc, "search ")
		if isConversational {
			taskMsg = task + "\n\n[INSTRUCTION: This is a conversational message. Respond directly using {\"status\":\"complete\",\"summary\":\"<your reply>\"}. Do NOT call any tool unless the message explicitly requires one.]"
		}
		messages = append(messages, Message{Role: "user", Content: taskMsg})
		chatRecorded := false
		for i := 0; i < MAX_LOOPS; i++ {
			response, err := chatWithMainLLM(messages)
			if err != nil {
				auditLog("ERROR", fmt.Sprintf("Main LLM Error: %v", err))
				mu.Lock()
				chatMessages = append(chatMessages, ChatMessage{
					Role:      "agent",
					Content:   fmt.Sprintf("I ran into an error: %v", err),
					Timestamp: time.Now(),
				})
				mu.Unlock()
				chatRecorded = true
				break
			}
			response = strings.TrimSpace(response)
			response = strings.TrimPrefix(response, "```json")
			response = strings.TrimPrefix(response, "```")
			response = strings.TrimSuffix(response, "```")
			response = strings.TrimSpace(response)
			var parsed map[string]interface{}
			if err := json.Unmarshal([]byte(response), &parsed); err != nil {
				auditLog("INFO", "Non-JSON response — using as direct chat reply")
				mu.Lock()
				chatMessages = append(chatMessages, ChatMessage{Role: "agent", Content: response, Timestamp: time.Now()})
				mu.Unlock()
				chatRecorded = true
				messages = append(messages, Message{Role: "assistant", Content: response})
				break
			}
			if status, ok := parsed["status"].(string); ok && status == "complete" {
				summary := ""
				if s, ok2 := parsed["summary"].(string); ok2 {
					summary = s
				}
				if summary == "" {
					auditLog("INFO", "Empty summary — requesting explicit reply from LLM")
					messages = append(messages, Message{Role: "assistant", Content: response})
					messages = append(messages, Message{Role: "user",
						Content: `Your summary was empty. Respond with {"status":"complete","summary":"<your actual reply to the user here>"}`})
					continue
				}
				mu.Lock()
				chatMessages = append(chatMessages, ChatMessage{Role: "agent", Content: summary, Timestamp: time.Now()})
				mu.Unlock()
				chatRecorded = true
				go func(taskText, sumText string) {
					mu.Lock()
					memoryDb = append(memoryDb, Memory{
						ID:          len(memoryDb) + 1,
						Tier:        Episodic,
						Content:     fmt.Sprintf("Task completed: %s — Result: %s", truncate(taskText, 100), truncate(sumText, 200)),
						Tags:        []string{"task_outcome", "episodic"},
						Importance:  5,
						AccessCount: 0,
						CreatedAt:   time.Now(),
						LastAccess:  time.Now(),
					})
					saveMemoryToFileLocked()
					mu.Unlock()
				}(task, summary)
				go func(t, s string) {
					key := t
					val := s
					tl := strings.ToLower(t + " " + s)
					nnCtrl.store(NN_DEFAULT, key, val)
					if containsAny(tl, []string{"code", "function", "bug", "error", "script", "program", "class", "module", "compile", "syntax", "api", "library"}) {
						nnCtrl.store(NN_CODE, key, val)
					}
					if containsAny(tl, []string{"prefer", "like", "want", "always", "never", "user", "name", "setting", "config", "fact", "know"}) {
						nnCtrl.store(NN_SEMANTIC, key, val)
					}
					if containsAny(tl, []string{"how to", "steps", "process", "workflow", "procedure", "sequence", "method", "pattern", "deploy", "install", "setup", "run"}) {
						nnCtrl.store(NN_PROCEDURAL, key, val)
					}
					auditLog("NEURAL", fmt.Sprintf("Task outcome stored across specialist networks"))
				}(truncate(task, 120), truncate(summary, 200))
				mu.Lock()
				for j := len(tasksDb) - 1; j >= 0; j-- {
					if tasksDb[j].Status == "running" {
						tasksDb[j].Status = "complete"
						tasksDb[j].Result = truncate(summary, 200)
						tasksDb[j].UpdatedAt = time.Now()
						saveTasksLocked()
						break
					}
				}
				mu.Unlock()
				auditLog("SUCCESS", "Task Complete")
				break
			}
			toolName, ok := parsed["tool"].(string)
			if !ok {
				messages = append(messages, Message{Role: "user", Content: "Invalid format. Need 'tool' and 'arguments'."})
				continue
			}
			toolParams, _ := parsed["arguments"].(map[string]interface{})
			toolFunc, exists := tools[toolName]
			if !exists {
				messages = append(messages, Message{Role: "user", Content: fmt.Sprintf("Tool %s not found.", toolName)})
				continue
			}
			result := toolFunc(toolParams)
			messages = append(messages, Message{Role: "assistant", Content: response})
			go learnFromLiveCall(truncate(task, 80), toolName)
			messages = append(messages, Message{Role: "user", Content: fmt.Sprintf("Tool Result: %s", result)})
		}
		if !chatRecorded {
			auditLog("INFO", "No chat response recorded — requesting forced reply")
			messages = append(messages, Message{Role: "user",
				Content: `Summarize what you did and respond to the user. Reply ONLY as valid JSON: {"status":"complete","summary":"<your reply to the user>"}`})
			if fr, ferr := chatWithMainLLM(messages); ferr == nil {
				fr = strings.TrimSpace(strings.TrimSuffix(strings.TrimPrefix(strings.TrimPrefix(strings.TrimSpace(fr), "```json"), "```"), "```"))
				var fp map[string]interface{}
				if json.Unmarshal([]byte(fr), &fp) == nil {
					if s, ok := fp["summary"].(string); ok && s != "" {
						mu.Lock()
						chatMessages = append(chatMessages, ChatMessage{Role: "agent", Content: s, Timestamp: time.Now()})
						mu.Unlock()
					}
				} else if len(fr) > 4 {
					mu.Lock()
					chatMessages = append(chatMessages, ChatMessage{Role: "agent", Content: fr, Timestamp: time.Now()})
					mu.Unlock()
				}
			}
		}
	}
}

func findRelevantMemories(task string) []string {
	taskLower := strings.ToLower(task)
	mu.Lock()
	mems := make([]Memory, len(memoryDb))
	copy(mems, memoryDb)
	mu.Unlock()

	type scored struct {
		content string
		score   int
	}
	var candidates []scored
	stopWords := map[string]bool{"the": true, "is": true, "at": true, "which": true,
		"on": true, "a": true, "an": true, "and": true, "to": true, "for": true, "of": true}
	var keywords []string
	for _, w := range strings.Fields(taskLower) {
		if !stopWords[w] && len(w) > 2 {
			keywords = append(keywords, w)
		}
	}
	for _, m := range mems {
		mLower := strings.ToLower(m.Content)
		hits := 0
		for _, kw := range keywords {
			if strings.Contains(mLower, kw) {
				hits++
			}
		}
		if hits == 0 {
			continue
		}
		tierBonus := map[MemoryTier]int{Procedural: 3, Semantic: 2, Episodic: 1}
		score := hits*m.Importance + tierBonus[m.Tier]
		candidates = append(candidates, scored{
			content: fmt.Sprintf("[%s] %s", strings.ToUpper(string(m.Tier)), m.Content),
			score:   score,
		})
	}
	for i := 1; i < len(candidates); i++ {
		for j := i; j > 0 && candidates[j].score > candidates[j-1].score; j-- {
			candidates[j], candidates[j-1] = candidates[j-1], candidates[j]
		}
	}
	var relevant []string
	max := 8
	if len(candidates) < max {
		max = len(candidates)
	}
	for _, c := range candidates[:max] {
		relevant = append(relevant, "- "+c.content)
	}
	return relevant
}

// --- MODULE SEARCH HELPERS ---
type ModuleResult struct {
	Path     string `json:"path"`
	Synopsis string `json:"synopsis"`
	Version  string `json:"version"`
	License  string `json:"license"`
}

type InstalledModule struct {
	Path    string `json:"path"`
	Version string `json:"version"`
}

func searchGoPackages(query string) ([]ModuleResult, error) {
	url := "https://pkg.go.dev/search?q=" + strings.ReplaceAll(query, " ", "+") + "&m=package"
	client := &http.Client{Timeout: 15 * time.Second}
	resp, err := client.Get(url)
	if err != nil {
		return nil, fmt.Errorf("search request failed: %v", err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	html := string(body)

	var results []ModuleResult
	searchStart := `<div class="SearchSnippet"`
	pos := 0
	for {
		idx := strings.Index(html[pos:], searchStart)
		if idx < 0 || len(results) >= 12 {
			break
		}
		pos += idx
		end := strings.Index(html[pos+len(searchStart):], searchStart)
		var block string
		if end < 0 {
			block = html[pos:]
		} else {
			block = html[pos : pos+len(searchStart)+end]
		}
		pos += len(searchStart)

		r := ModuleResult{}
		if hIdx := strings.Index(block, `href="/`); hIdx >= 0 {
			sub := block[hIdx+7:]
			if eIdx := strings.IndexAny(sub, `"?#`); eIdx >= 0 {
				candidate := sub[:eIdx]
				if strings.Contains(candidate, ".") && !strings.HasPrefix(candidate, "?") &&
					candidate != "pkg.go.dev" && !strings.HasPrefix(candidate, "about") &&
					!strings.HasPrefix(candidate, "account") && !strings.HasPrefix(candidate, "search") {
					r.Path = candidate
				}
			}
		}
		if r.Path == "" {
			continue
		}
		synTag := `SearchSnippet-synopsis`
		if sIdx := strings.Index(block, synTag); sIdx >= 0 {
			sub := block[sIdx:]
			if oIdx := strings.Index(sub, ">"); oIdx >= 0 {
				sub = sub[oIdx+1:]
				if cIdx := strings.Index(sub, "<"); cIdx >= 0 {
					r.Synopsis = strings.TrimSpace(sub[:cIdx])
				}
			}
		}
		if vIdx := strings.Index(block, `Version`); vIdx >= 0 {
			sub := block[vIdx:]
			if oIdx := strings.Index(sub, ">"); oIdx >= 0 {
				sub = sub[oIdx+1:]
				if cIdx := strings.Index(sub, "<"); cIdx >= 0 {
					v := strings.TrimSpace(sub[:cIdx])
					if strings.HasPrefix(v, "v") {
						r.Version = v
					}
				}
			}
		}
		if lIdx := strings.Index(block, `License`); lIdx >= 0 {
			sub := block[lIdx:]
			if oIdx := strings.Index(sub, ">"); oIdx >= 0 {
				sub = sub[oIdx+1:]
				if cIdx := strings.Index(sub, "<"); cIdx >= 0 {
					r.License = strings.TrimSpace(sub[:cIdx])
				}
			}
		}
		results = append(results, r)
	}
	return results, nil
}

func getInstalledModules() []InstalledModule {
	data, err := os.ReadFile("go.mod")
	if err != nil {
		return []InstalledModule{}
	}
	var mods []InstalledModule
	lines := strings.Split(string(data), "\n")
	inRequire := false
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "require (" {
			inRequire = true
			continue
		}
		if inRequire && line == ")" {
			inRequire = false
			continue
		}
		if inRequire || strings.HasPrefix(line, "require ") {
			parts := strings.Fields(strings.TrimPrefix(line, "require "))
			if len(parts) >= 2 {
				mods = append(mods, InstalledModule{Path: parts[0], Version: parts[1]})
			}
		}
	}
	return mods
}

// ─── WEB SEARCH TOOL ────────────────────────────────────────────────────────
type SearchResult struct {
	Title   string `json:"title"`
	URL     string `json:"url"`
	Snippet string `json:"snippet"`
}

func webSearch(params map[string]interface{}) string {
	query, _ := params["query"].(string)
	if query == "" {
		return "Error: 'query' is required."
	}
	mu.Lock()
	provider := webSearchProvider
	key := webSearchKey
	cx := webSearchCX
	mu.Unlock()

	auditLog("WEBSEARCH", fmt.Sprintf("Searching [%s]: %s", provider, query))

	var results []SearchResult
	var err error

	switch strings.ToLower(provider) {
	case "google":
		results, err = googleSearch(query, key, cx)
	case "brave":
		results, err = braveSearch(query, key)
	default:
		results, err = duckDuckGoSearch(query)
	}

	if err != nil {
		auditLog("ERROR", fmt.Sprintf("Web search failed: %v", err))
		return fmt.Sprintf("Search failed: %v", err)
	}
	if len(results) == 0 {
		return "No results found."
	}
	var lines []string
	for i, r := range results {
		lines = append(lines, fmt.Sprintf("[%d] %s\n    %s\n    %s", i+1, r.Title, r.URL, r.Snippet))
	}
	auditLog("WEBSEARCH", fmt.Sprintf("Got %d results for: %s", len(results), query))
	return strings.Join(lines, "\n\n")
}

func duckDuckGoSearch(query string) ([]SearchResult, error) {
	url := "https://html.duckduckgo.com/html/?q=" + strings.ReplaceAll(query, " ", "+")
	client := &http.Client{Timeout: 10 * time.Second}
	req, _ := http.NewRequest("GET", url, nil)
	req.Header.Set("User-Agent", "Mozilla/5.0 (compatible; AgentBot/1.0)")
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	html := string(body)

	var results []SearchResult
	marker := `class="result__title"`
	pos := 0
	for len(results) < 8 {
		idx := strings.Index(html[pos:], marker)
		if idx < 0 {
			break
		}
		pos += idx + len(marker)
		tStart := strings.Index(html[pos:], ">")
		if tStart < 0 {
			continue
		}
		tSub := html[pos+tStart+1:]
		tEnd := strings.Index(tSub, "</a>")
		title := ""
		if tEnd >= 0 {
			title = stripHTML(tSub[:tEnd])
		}
		urlMarker := `class="result__url"`
		uIdx := strings.Index(html[pos:], urlMarker)
		href := ""
		if uIdx >= 0 {
			uSub := html[pos+uIdx:]
			uTagEnd := strings.Index(uSub, ">")
			if uTagEnd >= 0 {
				uText := uSub[uTagEnd+1:]
				uClose := strings.Index(uText, "<")
				if uClose >= 0 {
					href = "https://" + strings.TrimSpace(uText[:uClose])
				}
			}
		}
		snipMarker := `class="result__snippet"`
		sIdx := strings.Index(html[pos:], snipMarker)
		snippet := ""
		if sIdx >= 0 {
			sSub := html[pos+sIdx:]
			sTagEnd := strings.Index(sSub, ">")
			if sTagEnd >= 0 {
				sText := sSub[sTagEnd+1:]
				sClose := strings.Index(sText, "</a>")
				if sClose >= 0 {
					snippet = stripHTML(sText[:sClose])
				}
			}
		}
		if title != "" {
			results = append(results, SearchResult{Title: title, URL: href, Snippet: snippet})
		}
	}
	return results, nil
}

func googleSearch(query, key, cx string) ([]SearchResult, error) {
	if key == "" || cx == "" {
		return nil, fmt.Errorf("Google API key and Custom Search Engine ID required")
	}
	url := fmt.Sprintf("https://www.googleapis.com/customsearch/v1?key=%s&cx=%s&q=%s&num=8",
		key, cx, strings.ReplaceAll(query, " ", "+"))
	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Get(url)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	var data struct {
		Items []struct {
			Title   string `json:"title"`
			Link    string `json:"link"`
			Snippet string `json:"snippet"`
		} `json:"items"`
	}
	if err := json.Unmarshal(body, &data); err != nil {
		return nil, err
	}
	var results []SearchResult
	for _, item := range data.Items {
		results = append(results, SearchResult{Title: item.Title, URL: item.Link, Snippet: item.Snippet})
	}
	return results, nil
}

func braveSearch(query, key string) ([]SearchResult, error) {
	if key == "" {
		return nil, fmt.Errorf("Brave Search API key required")
	}
	url := "https://api.search.brave.com/res/v1/web/search?q=" + strings.ReplaceAll(query, " ", "+") + "&count=8"
	client := &http.Client{Timeout: 10 * time.Second}
	req, _ := http.NewRequest("GET", url, nil)
	req.Header.Set("Accept", "application/json")
	req.Header.Set("X-Subscription-Token", key)
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	var data struct {
		Web struct {
			Results []struct {
				Title       string `json:"title"`
				URL         string `json:"url"`
				Description string `json:"description"`
			} `json:"results"`
		} `json:"web"`
	}
	if err := json.Unmarshal(body, &data); err != nil {
		return nil, err
	}
	var results []SearchResult
	for _, r := range data.Web.Results {
		results = append(results, SearchResult{Title: r.Title, URL: r.URL, Snippet: r.Description})
	}
	return results, nil
}

func stripHTML(s string) string {
	var b strings.Builder
	inTag := false
	for _, c := range s {
		if c == '<' {
			inTag = true
			continue
		}
		if c == '>' {
			inTag = false
			continue
		}
		if !inTag {
			b.WriteRune(c)
		}
	}
	out := b.String()
	out = strings.ReplaceAll(out, "&amp;", "&")
	out = strings.ReplaceAll(out, "&lt;", "<")
	out = strings.ReplaceAll(out, "&gt;", ">")
	out = strings.ReplaceAll(out, "&quot;", "\"")
	out = strings.ReplaceAll(out, "&#39;", "'")
	out = strings.ReplaceAll(out, "&nbsp;", " ")
	return strings.TrimSpace(out)
}

func fetchURL(params map[string]interface{}) string {
	url, _ := params["url"].(string)
	if url == "" {
		return "Error: 'url' is required."
	}
	auditLog("WEBSEARCH", fmt.Sprintf("Fetching URL: %s", url))
	client := &http.Client{Timeout: 15 * time.Second}
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return fmt.Sprintf("Error: %v", err)
	}
	req.Header.Set("User-Agent", "Mozilla/5.0 (compatible; AgentBot/1.0)")
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Sprintf("Fetch failed: %v", err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 32000))
	return truncate(stripHTML(string(body)), 4000)
}

// ─── SOCIAL TOOLS ────────────────────────────────────────────────────────────
func sendDiscord(params map[string]interface{}) string {
	msg, _ := params["message"].(string)
	if msg == "" {
		return "Error: 'message' required."
	}
	mu.Lock()
	token := discordToken
	channelID := discordChannelID
	mu.Unlock()
	if token == "" || channelID == "" {
		return "Discord not configured. Set token and channel ID in the Social tab."
	}
	url := fmt.Sprintf("https://discord.com/api/v10/channels/%s/messages", channelID)
	payload, _ := json.Marshal(map[string]string{"content": msg})
	req, _ := http.NewRequest("POST", url, bytes.NewBuffer(payload))
	req.Header.Set("Authorization", "Bot "+token)
	req.Header.Set("Content-Type", "application/json")
	resp, err := (&http.Client{Timeout: 10 * time.Second}).Do(req)
	if err != nil {
		return fmt.Sprintf("Discord error: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		auditLog("SOCIAL", fmt.Sprintf("Discord message sent to channel %s", channelID))
		return "Discord message sent."
	}
	b, _ := io.ReadAll(resp.Body)
	return fmt.Sprintf("Discord error %d: %s", resp.StatusCode, string(b))
}

func sendTelegram(params map[string]interface{}) string {
	msg, _ := params["message"].(string)
	if msg == "" {
		return "Error: 'message' required."
	}
	mu.Lock()
	token := telegramToken
	chatID := telegramChatID
	mu.Unlock()
	if token == "" || chatID == "" {
		return "Telegram not configured. Set token and chat ID in the Social tab."
	}
	url := fmt.Sprintf("https://api.telegram.org/bot%s/sendMessage", token)
	payload, _ := json.Marshal(map[string]string{"chat_id": chatID, "text": msg, "parse_mode": "Markdown"})
	resp, err := (&http.Client{Timeout: 10 * time.Second}).Post(url, "application/json", bytes.NewBuffer(payload))
	if err != nil {
		return fmt.Sprintf("Telegram error: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode == 200 {
		auditLog("SOCIAL", fmt.Sprintf("Telegram message sent to chat %s", chatID))
		return "Telegram message sent."
	}
	b, _ := io.ReadAll(resp.Body)
	return fmt.Sprintf("Telegram error %d: %s", resp.StatusCode, string(b))
}

func sendSlack(params map[string]interface{}) string {
	msg, _ := params["message"].(string)
	if msg == "" {
		return "Error: 'message' required."
	}
	mu.Lock()
	token := slackToken
	channel := slackChannel
	mu.Unlock()
	if token == "" || channel == "" {
		return "Slack not configured. Set token and channel in the Social tab."
	}
	payload, _ := json.Marshal(map[string]string{"channel": channel, "text": msg})
	req, _ := http.NewRequest("POST", "https://slack.com/api/chat.postMessage", bytes.NewBuffer(payload))
	req.Header.Set("Authorization", "Bearer "+token)
	req.Header.Set("Content-Type", "application/json")
	resp, err := (&http.Client{Timeout: 10 * time.Second}).Do(req)
	if err != nil {
		return fmt.Sprintf("Slack error: %v", err)
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	var result map[string]interface{}
	json.Unmarshal(b, &result)
	if ok, _ := result["ok"].(bool); ok {
		auditLog("SOCIAL", fmt.Sprintf("Slack message sent to %s", channel))
		return "Slack message sent."
	}
	return fmt.Sprintf("Slack error: %s", string(b))
}

func sendTwitter(params map[string]interface{}) string {
	msg, _ := params["message"].(string)
	if msg == "" {
		return "Error: 'message' required."
	}
	mu.Lock()
	consumerKey := twitterKey
	consumerSecret := twitterSecret
	accessToken := twitterToken
	accessSecret := twitterTokenSec
	mu.Unlock()
	if consumerKey == "" || accessToken == "" {
		return "Twitter/X not configured. Set keys in the Social tab."
	}
	payload, _ := json.Marshal(map[string]string{"text": msg})
	req, _ := http.NewRequest("POST", "https://api.twitter.com/2/tweets", bytes.NewBuffer(payload))
	req.Header.Set("Content-Type", "application/json")
	authHeader := buildOAuth1Header("POST", "https://api.twitter.com/2/tweets",
		consumerKey, consumerSecret, accessToken, accessSecret)
	req.Header.Set("Authorization", authHeader)
	resp, err := (&http.Client{Timeout: 10 * time.Second}).Do(req)
	if err != nil {
		return fmt.Sprintf("Twitter error: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode == 201 {
		auditLog("SOCIAL", "Tweet posted successfully")
		return "Tweet posted."
	}
	b, _ := io.ReadAll(resp.Body)
	return fmt.Sprintf("Twitter error %d: %s", resp.StatusCode, string(b))
}

func sendWhatsApp(params map[string]interface{}) string {
	msg, _ := params["message"].(string)
	if msg == "" {
		return "Error: 'message' required."
	}
	mu.Lock()
	sid := whatsAppSID
	token := whatsAppToken
	from := whatsAppFrom
	to := whatsAppTo
	mu.Unlock()
	if sid == "" || token == "" {
		return "WhatsApp not configured. Set Twilio credentials in the Social tab."
	}
	url := fmt.Sprintf("https://api.twilio.com/2010-04-01/Accounts/%s/Messages.json", sid)
	data := fmt.Sprintf("From=whatsapp:%s&To=whatsapp:%s&Body=%s",
		strings.ReplaceAll(from, "+", "%2B"),
		strings.ReplaceAll(to, "+", "%2B"),
		strings.ReplaceAll(msg, " ", "+"))
	req, _ := http.NewRequest("POST", url, strings.NewReader(data))
	req.SetBasicAuth(sid, token)
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	resp, err := (&http.Client{Timeout: 10 * time.Second}).Do(req)
	if err != nil {
		return fmt.Sprintf("WhatsApp error: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode == 201 {
		auditLog("SOCIAL", fmt.Sprintf("WhatsApp message sent to %s", to))
		return "WhatsApp message sent."
	}
	b, _ := io.ReadAll(resp.Body)
	return fmt.Sprintf("WhatsApp error %d: %s", resp.StatusCode, string(b))
}

func buildOAuth1Header(method, apiURL, consumerKey, consumerSecret, token, tokenSecret string) string {
	ts := fmt.Sprintf("%d", time.Now().Unix())
	nonce := fmt.Sprintf("%d", time.Now().UnixNano())
	params := fmt.Sprintf(
		`oauth_consumer_key="%s",oauth_nonce="%s",oauth_signature_method="HMAC-SHA1",`+
			`oauth_timestamp="%s",oauth_token="%s",oauth_version="1.0"`,
		consumerKey, nonce, ts, token)
	return "OAuth " + params
}

func pushLLMHistory(slot, url, model, key string, timeout int) {
	if url == "" || model == "" {
		return
	}
	label := model + " @ " + url
	profile := LLMProfile{Label: label, URL: url, Model: model, Key: key, Timeout: timeout}
	hist := llmHistory[slot]
	var filtered []LLMProfile
	for _, h := range hist {
		if h.URL != url || h.Model != model {
			filtered = append(filtered, h)
		}
	}
	filtered = append([]LLMProfile{profile}, filtered...)
	if len(filtered) > 5 {
		filtered = filtered[:5]
	}
	llmHistory[slot] = filtered
}

func autoClassifyMemory(content, tags string) MemoryTier {
	text := strings.ToLower(content + " " + tags)
	proceduralKeywords := []string{"how to", "steps", "process", "workflow", "procedure",
		"algorithm", "method", "pattern", "sequence", "run ", "execute", "compile", "build"}
	for _, kw := range proceduralKeywords {
		if strings.Contains(text, kw) {
			return Procedural
		}
	}
	episodicKeywords := []string{"completed", "happened", "today", "yesterday", "task",
		"conversation", "said", "asked", "told", "result", "outcome", "event", "session"}
	for _, kw := range episodicKeywords {
		if strings.Contains(text, kw) {
			return Episodic
		}
	}
	return Semantic
}

func consolidateMemory() {
	mu.Lock()
	defer mu.Unlock()
	if len(memoryDb) < 50 {
		return
	}
	cutoff := time.Now().AddDate(0, 0, -30)
	var kept []Memory
	seen := map[string]bool{}
	for _, m := range memoryDb {
		if m.Expires != nil && time.Now().After(*m.Expires) {
			continue
		}
		if m.Tier == Episodic && m.Importance < 4 && m.CreatedAt.Before(cutoff) {
			continue
		}
		key := string(m.Tier) + ":" + m.Content
		if len(key) > 80 {
			key = key[:80]
		}
		if m.Tier == Semantic && seen[key] {
			continue
		}
		seen[key] = true
		kept = append(kept, m)
	}
	if len(kept) < len(memoryDb) {
		auditLog("MEMORY", fmt.Sprintf("Consolidated: %d → %d memories", len(memoryDb), len(kept)))
		memoryDb = kept
		saveMemoryToFileLocked()
	}
}

// --- HELPERS ---
func containsAny(s string, keywords []string) bool {
	for _, kw := range keywords {
		if strings.Contains(s, kw) {
			return true
		}
	}
	return false
}

func maskKey(k string) string {
	if k == "" {
		return ""
	}
	if len(k) > 8 {
		return k[:4] + strings.Repeat("*", len(k)-8) + k[len(k)-4:]
	}
	return strings.Repeat("*", len(k))
}

func isAllAsterisks(s string) bool {
	for _, c := range s {
		if c != '*' {
			return false
		}
	}
	return true
}

// --- WEB SERVER ---
const HTML_TEMPLATE = `<!DOCTYPE html>
<html>
<head>
    <title>__TITLE__</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: 'Segoe UI', sans-serif; background: #1a1a1a; color: #d4d4d4; height: 100vh; display: flex; flex-direction: column; overflow: hidden; }

        /* ── HEADER ── */
        .header { background: #252526; border-bottom: 1px solid #333; padding: 8px 16px; display: flex; align-items: center; gap: 10px; flex-shrink: 0; }
        .header h1 { font-size: 1rem; color: #fff; letter-spacing: 0.05em; white-space: nowrap; }
        .header-badges { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; flex: 1; }
        .status-dot { height: 9px; width: 9px; background: #4ec9b0; border-radius: 50%; display: inline-block; animation: pulse 2s infinite; flex-shrink: 0; }
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }
        .header-badge { font-size: 0.68rem; padding: 2px 7px; border-radius: 10px; font-weight: bold; white-space: nowrap; }
        .hb-main      { background: #0e3a5a; color: #4fc1ff; border: 1px solid #1177bb; }
        .hb-coding    { background: #2d1a4a; color: #c586c0; border: 1px solid #5a2d82; }
        .hb-content   { background: #1a3a2a; color: #44aa88; border: 1px solid #2a6a4a; }
        .hb-media     { background: #3a1a3a; color: #c586c0; border: 1px solid #5a2d82; }
        .hb-off       { background: #2a2a2a; color: #555;    border: 1px solid #444; }
        .hb-root      { background: #3a1a1a; color: #f44747; border: 1px solid #7a1e1e; }
        .hb-task-pend { background: #3a2a1a; color: #ce9178; border: 1px solid #7a4a1e; }
        .hb-task-run  { background: #0e2a3a; color: #4fc1ff; border: 1px solid #1177bb; }

        /* ── TOP TAB BAR ── */
        .top-tabs { display: flex; background: #2d2d2d; border-bottom: 1px solid #3a3a3a; flex-shrink: 0; overflow-x: auto; scrollbar-width: none; }
        .top-tabs::-webkit-scrollbar { display: none; }
        .ttab { padding: 10px 16px; cursor: pointer; font-size: 0.78rem; color: #888; border-bottom: 2px solid transparent; white-space: nowrap; transition: all 0.15s; user-select: none; flex-shrink: 0; }
        .ttab:hover { color: #ccc; background: #333; }
        .ttab.active { color: #4ec9b0; border-bottom-color: #4ec9b0; background: #252526; }

        /* ── CONTENT AREA ── */
        .tab-content { flex: 1; overflow: hidden; display: flex; flex-direction: column; background: #1e1e1e; }
        .tab-pane { display: none; flex: 1; flex-direction: column; overflow: hidden; padding: 14px; }
        .tab-pane.active { display: flex; }

        /* ── BUTTONS ── */
        .btn { border: none; padding: 7px 14px; border-radius: 4px; cursor: pointer; font-weight: bold; font-size: 0.82rem; transition: background 0.15s; }
        .btn-blue   { background: #0e639c; color: #fff; } .btn-blue:hover   { background: #1177bb; }
        .btn-green  { background: #2d6e2d; color: #fff; } .btn-green:hover  { background: #3a8a3a; }
        .btn-amber  { background: #7a4a1e; color: #fff; } .btn-amber:hover  { background: #9a5e2a; }
        .btn-purple { background: #5a2d82; color: #fff; } .btn-purple:hover { background: #6e3a9a; }
        .btn-teal   { background: #0e5a5a; color: #fff; } .btn-teal:hover   { background: #0e7070; }
        .btn-red    { background: #7a1e1e; color: #fff; } .btn-red:hover    { background: #9a2a2a; }

        /* ── CHAT PANE ── */
        .chat-wrap { display: flex; flex-direction: column; flex: 1; overflow: hidden; }
        .chat-hdr { padding: 9px 16px; border-bottom: 1px solid #333; display: flex; align-items: center; justify-content: space-between; flex-shrink: 0; background: #252526; border-radius: 8px 8px 0 0; }
        .chat-hdr h2 { font-size: 0.92rem; color: #fff; }
        .chat-body { flex: 1; display: flex; overflow: hidden; gap: 0; }
        .chat-messages { flex: 1; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 10px; background: #1e1e1e; }
        .msg-row { display: flex; gap: 10px; max-width: 85%; }
        .msg-row.user { align-self: flex-end; flex-direction: row-reverse; }
        .msg-row.agent { align-self: flex-start; }
        .msg-avatar { width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1rem; flex-shrink: 0; margin-top: 2px; }
        .msg-avatar.user-av  { background: #0e3a5a; }
        .msg-avatar.agent-av { background: #1a3a2a; }
        .msg-bubble { padding: 9px 13px; border-radius: 12px; font-size: 0.85rem; line-height: 1.55; word-break: break-word; max-width: 100%; }
        .msg-row.user  .msg-bubble { background: #0e3a5a; color: #d4eeff; border-radius: 12px 3px 12px 12px; }
        .msg-row.agent .msg-bubble { background: #252526; color: #d4d4d4; border: 1px solid #333; border-radius: 3px 12px 12px 12px; }
        .msg-time { font-size: 0.65rem; color: #555; margin-top: 3px; text-align: right; }
        .msg-row.agent .msg-time { text-align: left; }
        .chat-activity { width: 320px; flex-shrink: 0; border-left: 1px solid #333; display: flex; flex-direction: column; background: #1a1a1a; }
        .activity-hdr { padding: 8px 12px; border-bottom: 1px solid #333; font-size: 0.72rem; font-weight: bold; color: #666; text-transform: uppercase; letter-spacing: 0.06em; display: flex; align-items: center; justify-content: space-between; flex-shrink: 0; }
        .activity-toggle { font-size: 0.68rem; color: #4ec9b0; cursor: pointer; text-transform: none; letter-spacing: 0; }
        #activity-feed { flex: 1; overflow-y: auto; padding: 6px; font-family: monospace; font-size: 0.75rem; }
        .act-entry { margin-bottom: 3px; padding: 3px 7px; border-left: 2px solid #444; border-radius: 0 2px 2px 0; line-height: 1.35; word-break: break-word; color: #888; }
        .act-INFO        { border-color: #569cd6; color: #7aaad4; }
        .act-ACTION      { border-color: #dcdcaa; color: #b0a870; }
        .act-ERROR       { border-color: #f44747; color: #d47070; background: #2a1a1a; }
        .act-MEMORY      { border-color: #c586c0; color: #9a6a95; }
        .act-CONTEXT     { border-color: #4ec9b0; }
        .act-SUCCESS     { border-color: #6a9955; color: #6a9955; }
        .act-HEARTBEAT   { border-color: #555; color: #555; }
        .act-CONFIG      { border-color: #ce9178; color: #9a7058; }
        .act-SYSTEM      { border-color: #555; }
        .act-MODIFICATION{ border-color: #f44747; }
        .act-CODING_LLM  { border-color: #c586c0; background: #1e1830; }
        .act-CONTENT_LLM { border-color: #44aa88; background: #101a14; }
        .act-TASK        { border-color: #dcdcaa; background: #1a1a10; }
        .act-MODULE      { border-color: #4ec9b0; background: #0e1a1a; }
        .chat-footer { padding: 10px 14px; border-top: 1px solid #333; flex-shrink: 0; background: #252526; border-radius: 0 0 8px 8px; }
        .input-row { display: flex; gap: 8px; align-items: flex-end; }
        .chat-input-wrap { flex: 1; position: relative; }
        #user-input { width: 100%; resize: none; height: 42px; max-height: 120px; overflow-y: auto; line-height: 1.5; padding: 9px 12px; border-radius: 8px; font-size: 0.88rem; }
        .send-btn { width: 40px; height: 42px; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 1.1rem; padding: 0; flex-shrink: 0; }
        .log-entry { margin-bottom: 4px; padding: 5px 8px; border-left: 3px solid #555; background: #2a2a2a; border-radius: 0 3px 3px 0; line-height: 1.4; word-break: break-word; }
        .log-INFO        { border-color: #569cd6; }
        #chat-output .log-entry { font-size: 0.83rem; }
        .log-ACTION      { border-color: #dcdcaa; }
        .log-ERROR       { border-color: #f44747; }
        .log-NEURAL   { border-left: 3px solid #c586c0; }
        .log-SUCCESS  { border-left: 3px solid #6a9955; }
        .log-ACTION   { border-left: 3px solid #dcdcaa; }
        .audit-lvl    { min-width: 90px; display: inline-block; font-size: 0.72rem; }
        .btn-purple   { background: #3a1a4a; color: #c586c0; border: 1px solid #5a2d82; }
        @keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:0.3;} }
        .btn-purple:hover { background: #4a2a5a; }
        .log-MEMORY      { border-color: #c586c0; }
        .log-CONTEXT     { border-color: #4ec9b0; }
        .log-SUCCESS     { border-color: #6a9955; }
        .log-HEARTBEAT   { border-color: #aaa; background: #333; font-weight: bold; }
        .log-CONFIG      { border-color: #ce9178; }
        .log-SYSTEM      { border-color: #666; }
        .log-MODIFICATION{ border-color: #f44747; }
        .log-CODING_LLM  { border-color: #c586c0; background: #2d2040; }
        .log-CONTENT_LLM { border-color: #44aa88; background: #1a3a2a; }
        .log-TASK        { border-color: #dcdcaa; background: #2a2a18; }
        .log-MODULE      { border-color: #4ec9b0; background: #1a2a2a; }
        .empty-state { color: #555; font-size: 0.85rem; text-align: center; margin-top: 30px; }
        .chat-footer { padding: 10px 12px; border-top: 1px solid #333; flex-shrink: 0; }
        .input-row { display: flex; gap: 8px; }
        #user-input { flex: 1; }

        /* ── SHARED INPUTS ── */
        input[type="text"], input[type="password"], textarea, select {
            background: #3c3c3c; border: 1px solid #555; color: #d4d4d4;
            padding: 7px 10px; border-radius: 4px; font-family: inherit; font-size: 0.86rem;
        }
        input:focus, textarea:focus, select:focus { outline: none; border-color: #0e639c; }

        /* ── SCROLLABLE LISTS ── */
        .scroll-list { flex: 1; overflow-y: auto; }
        .search-bar { width: 100%; margin-bottom: 8px; flex-shrink: 0; }

        /* ── MEMORY ── */
        .mem-item { background: #2e2e2e; padding: 7px 9px; margin-bottom: 5px; border-radius: 3px; border-left: 2px solid #c586c0; font-size: 0.8rem; line-height: 1.4; }
        .mem-meta { font-size: 0.7rem; color: #777; margin-bottom: 3px; display: flex; justify-content: space-between; flex-wrap: wrap; gap: 4px; }
        .tag { font-size: 0.68rem; background: #444; padding: 1px 5px; border-radius: 3px; color: #aaa; }

        /* ── SESSIONS ── */
        .session-item { background: #2e2e2e; padding: 8px 10px; margin-bottom: 5px; border-radius: 3px; border-left: 2px solid #569cd6; cursor: pointer; font-size: 0.8rem; transition: background 0.1s; }
        .session-item:hover { background: #383838; }
        .session-title { color: #d4d4d4; font-weight: bold; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin-bottom: 3px; }
        .session-meta { font-size: 0.7rem; color: #777; }

        /* ── PERSONA ── */
        #personality-text { width: 100%; flex: 1; resize: none; font-size: 0.83rem; line-height: 1.55; padding: 9px; border-radius: 4px; }
        .personality-hint { font-size: 0.75rem; color: #666; margin-bottom: 8px; line-height: 1.6; flex-shrink: 0; }
        #personality-status { font-size: 0.75rem; color: #6a9955; margin-top: 5px; text-align: center; min-height: 1em; flex-shrink: 0; }

        /* ── AUDIT ── */
        .audit-entry { font-size: 0.78rem; padding: 4px 6px; margin-bottom: 3px; border-left: 2px solid #555; background: #2a2a2a; border-radius: 0 2px 2px 0; word-break: break-word; }

        /* ── LLM CONFIG PANELS ── */
        .llm-config-panel { display: flex; flex-direction: column; gap: 0; max-width: 600px; overflow-y: auto; }
        .field-group { margin-bottom: 11px; flex-shrink: 0; }
        .field-label { font-size: 0.75rem; color: #aaa; margin-bottom: 4px; display: block; font-weight: bold; }
        .field-hint  { font-size: 0.70rem; color: #555; margin-top: 3px; line-height: 1.4; }
        .field-input { width: 100%; }
        .llm-status-badge { display: inline-flex; align-items: center; gap: 5px; font-size: 0.72rem; padding: 3px 8px; border-radius: 10px; margin-bottom: 10px; flex-shrink: 0; }
        .badge-ok   { background: #1a3a1a; color: #6a9955; border: 1px solid #2d6e2d; }
        .badge-warn { background: #3a2a1a; color: #ce9178; border: 1px solid #7a4a1e; }
        .badge-dot  { width: 6px; height: 6px; border-radius: 50%; background: currentColor; flex-shrink: 0; }
        .llm-save-status { font-size: 0.75rem; color: #6a9955; margin-top: 6px; min-height: 1em; flex-shrink: 0; }
        .divider { border: none; border-top: 1px solid #333; margin: 10px 0; flex-shrink: 0; }
        .test-result { background: #1a2a1a; border: 1px solid #2d6e2d; border-radius: 4px; padding: 8px; font-size: 0.78rem; color: #6a9955; margin-top: 8px; max-height: 90px; overflow-y: auto; word-break: break-word; display: none; flex-shrink: 0; }
        .llm-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; width: 100%; }
        .llm-card { background: #252526; border: 1px solid #333; border-radius: 8px; padding: 14px; display: flex; flex-direction: column; }
        .llm-card-title { font-size: 0.82rem; font-weight: bold; color: #fff; margin-bottom: 10px; padding-bottom: 7px; border-bottom: 1px solid #333; }

        /* ── TASKS ── */
        .task-item { background: #2e2e2e; padding: 8px 10px; margin-bottom: 6px; border-radius: 4px; border-left: 3px solid #555; font-size: 0.8rem; line-height: 1.4; }
        .task-pending  { border-color: #ce9178; }
        .task-running  { border-color: #4fc1ff; background: #1a2a3a; }
        .task-complete { border-color: #6a9955; }
        .task-failed   { border-color: #f44747; background: #2a1a1a; }
        .task-title    { font-weight: bold; color: #d4d4d4; margin-bottom: 3px; }
        .task-desc     { color: #888; font-size: 0.75rem; margin-bottom: 3px; }
        .task-result   { color: #6a9955; font-size: 0.75rem; margin-top: 3px; font-style: italic; }
        .task-meta     { font-size: 0.70rem; color: #666; display: flex; justify-content: space-between; flex-wrap: wrap; gap: 4px; }
        .task-badge    { font-size: 0.68rem; padding: 1px 6px; border-radius: 3px; font-weight: bold; }
        .tb-pending  { background: #3a2a1a; color: #ce9178; }
        .tb-running  { background: #0e2a3a; color: #4fc1ff; }
        .tb-complete { background: #1a3a1a; color: #6a9955; }
        .tb-failed   { background: #3a1a1a; color: #f44747; }
        .tb-p1 { color: #f44747; } .tb-p2 { color: #888; } .tb-p3 { color: #555; }
        .task-add-form { background: #252526; border: 1px solid #3a3a3a; border-radius: 6px; padding: 12px; margin-bottom: 10px; flex-shrink: 0; max-width: 680px; }
        .task-actions { display: flex; gap: 5px; margin-top: 6px; }
        .task-act-btn { font-size: 0.68rem; padding: 2px 8px; border-radius: 3px; border: none; cursor: pointer; font-weight: bold; }
        .tact-edit    { background: #0e3a5a; color: #4fc1ff; } .tact-edit:hover    { background: #1177bb; }
        .tact-delete  { background: #5a0e0e; color: #f44747; } .tact-delete:hover  { background: #8a1a1a; }
        .tact-status  { background: #2d2d2d; color: #aaa;    border: 1px solid #444; } .tact-status:hover  { background: #444; }
        .task-edit-row { background: #1e2a1e; border: 1px solid #2d6e2d; border-radius: 6px; padding: 10px; margin-bottom: 6px; }
        .task-edit-row input, .task-edit-row textarea, .task-edit-row select { width: 100%; margin-bottom: 6px; font-size: 0.82rem; }
        .task-edit-row textarea { resize: none; height: 54px; }
        .task-edit-btns { display: flex; gap: 6px; }
        .tasks-wrap { display: flex; gap: 14px; flex: 1; overflow: hidden; }
        .tasks-main  { flex: 1; display: flex; flex-direction: column; overflow: hidden; min-width: 0; }
        .tasks-stats { width: 180px; flex-shrink: 0; display: flex; flex-direction: column; gap: 8px; }
        .stat-card { background: #252526; border: 1px solid #333; border-radius: 6px; padding: 10px; text-align: center; }
        .stat-num  { font-size: 1.6rem; font-weight: bold; line-height: 1; }
        .stat-label{ font-size: 0.70rem; color: #666; margin-top: 3px; }
        .stat-pend { color: #ce9178; } .stat-run  { color: #4fc1ff; }
        .stat-done { color: #6a9955; } .stat-fail { color: #f44747; }
        .task-add-form input, .task-add-form textarea, .task-add-form select { width: 100%; margin-bottom: 6px; }
        .task-add-form textarea { resize: none; height: 54px; }
        .task-filters { display: flex; gap: 5px; margin-bottom: 8px; flex-shrink: 0; flex-wrap: wrap; }
        .filter-btn { font-size: 0.70rem; padding: 3px 10px; border-radius: 10px; border: 1px solid #444; background: #2a2a2a; color: #777; cursor: pointer; }
        .filter-btn.active { background: #0e3a5a; color: #4fc1ff; border-color: #1177bb; }

        /* ── MODULES ── */
        .mod-search-row { display:flex; gap:6px; margin-bottom:10px; flex-shrink:0; max-width:600px; }
        .mod-search-row input { flex:1; }
        .mod-result { background:#2e2e2e; border:1px solid #3a3a3a; border-radius:4px; padding:8px 10px; margin-bottom:6px; font-size:0.8rem; line-height:1.5; }
        .mod-result:hover { border-color:#555; }
        .mod-name { font-weight:bold; color:#4fc1ff; font-size:0.85rem; word-break:break-all; }
        .mod-synopsis { color:#aaa; font-size:0.75rem; margin:2px 0 4px; }
        .mod-meta { font-size:0.70rem; color:#666; display:flex; gap:8px; flex-wrap:wrap; align-items:center; }
        .mod-ver { color:#ce9178; } .mod-license { color:#6a9955; }
        .mod-btn { font-size:0.70rem; padding:2px 8px; border-radius:3px; border:none; cursor:pointer; font-weight:bold; margin-left:auto; }
        .mod-btn-add { background:#0e5a0e; color:#6a9955; } .mod-btn-add:hover { background:#1a8a1a; }
        .mod-btn-rm  { background:#5a0e0e; color:#f44747; } .mod-btn-rm:hover  { background:#8a1a1a; }
        .installed-mod { background:#1a2a1a; border:1px solid #2d6e2d; border-radius:4px; padding:7px 10px; margin-bottom:5px; font-size:0.78rem; display:flex; align-items:center; gap:8px; }
        .installed-mod-path { color:#4ec9b0; flex:1; word-break:break-all; font-family:monospace; }
        .installed-mod-ver  { color:#ce9178; font-size:0.72rem; white-space:nowrap; }
        .mod-tabs { display:flex; gap:0; margin-bottom:10px; flex-shrink:0; border-bottom:1px solid #333; }
        .mod-tab { font-size:0.75rem; padding:6px 14px; cursor:pointer; color:#777; border-bottom:2px solid transparent; }
        .mod-tab.active { color:#4ec9b0; border-bottom-color:#4ec9b0; }
        .mod-status { font-size:0.75rem; min-height:1.2em; margin-bottom:6px; flex-shrink:0; }

        /* ── HOST ── */
        .host-panel { display:flex; flex-direction:column; gap:0; max-width:500px; }
        .warning-box { background:#2a1a1a; border:1px solid #7a1e1e; border-radius:4px; padding:9px 10px; margin-bottom:10px; font-size:0.75rem; color:#ce9178; line-height:1.55; flex-shrink:0; }
        .warning-box b { color:#f44747; }
        .verify-result { background:#1a1a2a; border:1px solid #569cd6; border-radius:4px; padding:8px; font-size:0.78rem; color:#4fc1ff; margin-top:8px; max-height:80px; overflow-y:auto; word-break:break-word; display:none; flex-shrink:0; font-family:monospace; }


        /* ── LLM HISTORY DROPDOWN ── */
        .llm-history-wrap { margin-bottom:10px; flex-shrink:0; }
        .llm-history-wrap label { font-size:0.70rem; color:#888; font-weight:bold; display:block; margin-bottom:4px; }
        .llm-history-select { width:100%; background:#1a1a1a; color:#ccc; border:1px solid #3a3a3a;
            border-radius:4px; padding:6px 8px; font-size:0.78rem; cursor:pointer; }
        .llm-history-select:focus { outline:none; border-color:#4ec9b0; }
        .llm-history-select option { background:#1e1e1e; color:#ccc; }

        /* ── DELEGATION TOGGLE ── */
        .delegation-bar { display:flex; align-items:center; justify-content:space-between;
            background:#1e2a1e; border:1px solid #2d6e2d; border-radius:6px;
            padding:10px 14px; margin-bottom:14px; flex-shrink:0; }
        .delegation-bar.disabled { background:#2a1a1a; border-color:#7a1e1e; }
        .delegation-bar-left { display:flex; flex-direction:column; gap:2px; }
        .delegation-bar-title { font-size:0.82rem; font-weight:bold; color:#fff; }
        .delegation-bar-desc  { font-size:0.70rem; color:#888; line-height:1.4; max-width:400px; }
        .toggle-wrap { display:flex; align-items:center; gap:8px; flex-shrink:0; }
        .toggle-label { font-size:0.75rem; font-weight:bold; }
        .toggle-label.on  { color:#6a9955; }
        .toggle-label.off { color:#f44747; }
        .toggle-switch { position:relative; width:44px; height:24px; cursor:pointer; flex-shrink:0; }
        .toggle-switch input { opacity:0; width:0; height:0; }
        .toggle-track { position:absolute; inset:0; background:#3c3c3c; border-radius:24px;
            transition:background 0.2s; border:1px solid #555; }
        .toggle-thumb { position:absolute; top:3px; left:3px; width:16px; height:16px;
            background:#888; border-radius:50%; transition:all 0.2s; }
        .toggle-switch input:checked ~ .toggle-track { background:#1a4a1a; border-color:#2d6e2d; }
        .toggle-switch input:checked ~ .toggle-thumb { left:23px; background:#6a9955; }

        /* ── WEB SEARCH TAB ── */
        .search-provider-grid { display:grid; grid-template-columns:1fr 1fr 1fr; gap:8px; margin-bottom:12px; }
        .provider-card { background:#2e2e2e; border:2px solid #3a3a3a; border-radius:6px; padding:10px; text-align:center; cursor:pointer; transition:all 0.15s; }
        .provider-card:hover { border-color:#555; }
        .provider-card.selected { border-color:#4ec9b0; background:#1a2a2a; }
        .provider-icon { font-size:1.5rem; margin-bottom:4px; }
        .provider-name { font-size:0.75rem; font-weight:bold; color:#aaa; }
        .search-test-box { background:#1a1a1a; border:1px solid #333; border-radius:4px; padding:10px; margin-top:8px; font-family:monospace; font-size:0.75rem; color:#888; max-height:200px; overflow-y:auto; white-space:pre-wrap; display:none; }
        /* ── SOCIAL TAB ── */
        .social-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:14px; overflow-y:auto; }
        .social-card { background:#252526; border:1px solid #333; border-radius:8px; padding:14px; display:flex; flex-direction:column; gap:0; }
        .social-card-hdr { display:flex; align-items:center; gap:8px; margin-bottom:10px; padding-bottom:8px; border-bottom:1px solid #333; }
        .social-icon { font-size:1.4rem; }
        .social-name { font-size:0.85rem; font-weight:bold; color:#fff; flex:1; }
        .social-status-dot { width:8px; height:8px; border-radius:50%; background:#555; flex-shrink:0; }
        .social-status-dot.on { background:#4ec9b0; animation:pulse 2s infinite; }
        .social-field { margin-bottom:7px; }
        .social-field label { font-size:0.70rem; color:#888; display:block; margin-bottom:3px; font-weight:bold; }
        .social-field input { width:100%; font-size:0.82rem; }
        .social-btns { display:flex; gap:6px; margin-top:8px; }
        .social-save-status { font-size:0.72rem; min-height:1.2em; margin-top:5px; }
        /* ── PERSONALITY BADGE ── */
        .personality-applied { font-size:0.70rem; color:#6a9955; background:#1a2a1a; border:1px solid #2d6e2d; border-radius:3px; padding:2px 7px; display:inline-block; margin-bottom:8px; }
        /* ── MODAL ── */
        .modal-overlay { display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.75); z-index: 200; align-items: center; justify-content: center; }
        .modal-overlay.open { display: flex; }
        .modal { background: #252526; border: 1px solid #444; border-radius: 8px; width: 680px; max-width: 95vw; max-height: 80vh; display: flex; flex-direction: column; overflow: hidden; }
        .modal-hdr { padding: 12px 15px; border-bottom: 1px solid #333; display: flex; align-items: center; justify-content: space-between; }
        .modal-hdr span { font-weight: bold; color: #fff; font-size: 0.9rem; }
        .modal-body { padding: 12px; overflow-y: auto; flex: 1; font-family: monospace; font-size: 0.83rem; }
        .close-btn { background: #3c3c3c; border: 1px solid #555; color: #aaa; padding: 4px 12px; border-radius: 3px; cursor: pointer; font-size: 0.82rem; }
        .close-btn:hover { background: #4a4a4a; }
    </style>
</head>
<body>

<div class="header">
    <span class="status-dot" title="Agent Active"></span>
    <h1 id="header-title">__TITLE__</h1>
    <div class="header-badges">
        <span id="hdr-main"    class="header-badge hb-off">Main: —</span>
        <span id="hdr-coding"  class="header-badge hb-off">Code: —</span>
        <span id="hdr-media"   class="header-badge hb-off">Media: —</span>
        <span id="hdr-content" class="header-badge hb-off">Content: —</span>
        <span id="hdr-tasks"   class="header-badge hb-off">Tasks: idle</span>
        <span id="hdr-root"    class="header-badge hb-off">Root: off</span>
    </div>
</div>

<div class="top-tabs">
    <div class="ttab active" onclick="switchTab('chat')">💬 Chat</div>
    <div class="ttab" onclick="switchTab('tasks')">📋 Tasks</div>
    <div class="ttab" onclick="switchTab('sessions')">📁 Sessions</div>
    <div class="ttab" onclick="switchTab('personality')">🎭 Persona</div>
    <div class="ttab" onclick="switchTab('main-llm')">⚙ Main LLM</div>
    <div class="ttab" onclick="switchTab('coding-llm')">💻 Code LLM</div>
    <div class="ttab" onclick="switchTab('media-llm')">🎬 Media LLM</div>
    <div class="ttab" onclick="switchTab('content-llm')">✍️ Content LLM</div>
    <div class="ttab" onclick="switchTab('websearch')">🔍 Web Search</div>
    <div class="ttab" onclick="switchTab('social')">📡 Social</div>
    <div class="ttab" onclick="switchTab('modules')">📦 Modules</div>
    <div class="ttab" onclick="switchTab('host')">🖥 Host</div>
    <div class="ttab" onclick="switchTab('neural')">🧠 Neural</div>
    <div class="ttab" onclick="switchTab('audit')">📜 Audit</div>
</div>

<div class="tab-content">

    <!-- ===== CHAT ===== -->
    <div id="pane-chat" class="tab-pane active">
        <div class="chat-wrap">
            <div class="chat-hdr">
                <h2>💬 Agent Chat</h2>
                <div style="display:flex;gap:8px;align-items:center;">
                    <span id="agent-typing" style="font-size:0.72rem;color:#4ec9b0;display:none;">Agent is working...</span>
                    <button class="btn btn-amber" onclick="newChat()" style="padding:5px 12px;">+ New Chat</button>
                </div>
            </div>
            <div class="chat-body">
                <div class="chat-messages" id="chat-messages">
                    <div class="empty-state" id="chat-empty" style="margin:auto;">Send a message to get started.</div>
                </div>
                <div class="chat-activity" id="chat-activity-panel">
                    <div class="activity-hdr">
                        <span>Live Activity</span>
                        <span class="activity-toggle" onclick="toggleActivity()">hide</span>
                    </div>
                    <div id="activity-feed"></div>
                </div>
            </div>
            <div class="chat-footer">
                <div class="input-row">
                    <div class="chat-input-wrap">
                        <textarea id="user-input" placeholder="Give the agent a task... (Enter to send, Shift+Enter for newline)" rows="1"></textarea>
                    </div>
                    <button class="btn btn-blue send-btn" onclick="sendTask()" title="Send">➤</button>
                </div>
            </div>
        </div>
    </div>


    <!-- ===== TASKS ===== -->
    <div id="pane-tasks" class="tab-pane">
        <div class="tasks-wrap">
            <div class="tasks-main">
                <div class="task-add-form">
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:6px;">
                        <input type="text" id="task-title" placeholder="Task title (required)" style="grid-column:1/-1;">
                        <textarea id="task-desc" placeholder="Description (optional)" style="grid-column:1/-1;resize:none;height:52px;"></textarea>
                        <select id="task-priority">
                            <option value="1">🔴 High priority</option>
                            <option value="2" selected>🟡 Normal priority</option>
                            <option value="3">🟢 Low priority</option>
                        </select>
                        <button class="btn btn-blue" onclick="addTask()">+ Add Task</button>
                    </div>
                    <div id="task-add-status" style="font-size:0.73rem;color:#6a9955;min-height:1em;"></div>
                </div>
                <div class="task-filters">
                    <span class="filter-btn active" onclick="setTaskFilter('all',this)">All</span>
                    <span class="filter-btn" onclick="setTaskFilter('pending',this)">Pending</span>
                    <span class="filter-btn" onclick="setTaskFilter('running',this)">Running</span>
                    <span class="filter-btn" onclick="setTaskFilter('complete',this)">Done</span>
                    <span class="filter-btn" onclick="setTaskFilter('failed',this)">Failed</span>
                </div>
                <div class="scroll-list" id="task-list"><div class="empty-state">No tasks yet.</div></div>
            </div>
            <div class="tasks-stats">
                <div class="stat-card"><div class="stat-num stat-pend" id="stat-pending">0</div><div class="stat-label">Pending</div></div>
                <div class="stat-card"><div class="stat-num stat-run"  id="stat-running">0</div><div class="stat-label">Running</div></div>
                <div class="stat-card"><div class="stat-num stat-done" id="stat-complete">0</div><div class="stat-label">Complete</div></div>
                <div class="stat-card"><div class="stat-num stat-fail" id="stat-failed">0</div><div class="stat-label">Failed</div></div>
            </div>
        </div>
    </div>

    <!-- ===== SESSIONS ===== -->
    <div id="pane-sessions" class="tab-pane">
        <div style="font-size:0.80rem;color:#666;margin-bottom:10px;flex-shrink:0;">Click a session to view its full log.</div>
        <div class="scroll-list" id="session-list"><div class="empty-state">No sessions yet.</div></div>
    </div>

    <!-- ===== PERSONA ===== -->
    <div id="pane-personality" class="tab-pane">
        <div style="max-width:600px;display:flex;flex-direction:column;flex:1;overflow:hidden;">
            <div style="margin-bottom:10px;flex-shrink:0;">
                <label class="field-label" for="ui-title-input">Agent Name (header title)</label>
                <div style="display:flex;gap:6px;">
                    <input type="text" id="ui-title-input" placeholder="Lily" style="flex:1;">
                    <button class="btn btn-green" onclick="saveUITitle()" style="padding:7px 14px;">Save</button>
                </div>
                <div id="ui-title-status" style="font-size:0.73rem;color:#6a9955;margin-top:4px;min-height:1em;"></div>
            </div>
            <hr class="divider">
            <div class="personality-applied">✓ Personality is applied to Main LLM, Coding LLM, Media LLM, and Content LLM</div>
            <div class="personality-hint">
                Define how the agent thinks and behaves. Changes take effect on the next task.<br><br>
                <b style="color:#aaa">Examples:</b> Always explain reasoning step by step · Prefer Python · Be concise · Ask before acting
            </div>
            <textarea id="personality-text" placeholder="Enter personality traits, one per line..."></textarea>
            <button class="btn btn-green" style="margin-top:8px;width:100%;padding:9px" onclick="savePersonality()">Save Personality</button>
            <div id="personality-status"></div>
        </div>
    </div>

    <!-- ===== MAIN LLM ===== -->
    <div id="pane-main-llm" class="tab-pane">
        <div class="llm-config-panel">

            <div class="llm-history-wrap">
                <label>⏱ Recent LLMs</label>
                <select class="llm-history-select" id="hist-main" onchange="applyLLMHistory('main',this.value)">
                    <option value="">— select a saved profile —</option>
                </select>
            </div>
            <div id="main-llm-badge" class="llm-status-badge badge-warn"><span class="badge-dot"></span><span id="main-llm-badge-text">Loading...</span></div>
            <div class="field-group">
                <label class="field-label" for="main-llm-url">Base URL</label>
                <input type="text" id="main-llm-url" class="field-input" placeholder="http://localhost:11434">
                <div class="field-hint">Root address — <code>/v1/chat/completions</code> appended automatically. Works with Ollama, LM Studio, vLLM, OpenAI, Groq, etc.</div>
            </div>
            <div class="field-group">
                <label class="field-label" for="main-llm-model">Model Name</label>
                <input type="text" id="main-llm-model" class="field-input" placeholder="llama3 / gpt-4o / mistral">
            </div>
            <div class="field-group">
                <label class="field-label" for="main-llm-key">API Key</label>
                <input type="password" id="main-llm-key" class="field-input" placeholder="sk-... (leave blank if not required)">
                <div class="field-hint">Sent as <code>Authorization: Bearer &lt;key&gt;</code>.</div>
            </div>
            <div class="field-group">
                <label class="field-label" for="main-llm-timeout">Request Timeout (seconds)</label>
                <input type="number" id="main-llm-timeout" class="field-input" placeholder="300" min="10" max="3600">
                <div class="field-hint">How long to wait for a response. Default 300s. Increase for slow/large models. Retries up to 2× on timeout.</div>
            </div>
            <hr class="divider">
            <button class="btn btn-green" style="width:100%;padding:9px;margin-bottom:6px" onclick="saveMainLLM()">💾 Save</button>
            <button class="btn btn-teal"  style="width:100%;padding:9px" onclick="testMainLLM()">⚡ Test Connection</button>
            <div id="main-llm-status" class="llm-save-status"></div>
            <div id="main-llm-test-result" class="test-result"></div>
        </div>
    </div>

    <!-- ===== CODING LLM ===== -->
    <div id="pane-coding-llm" class="tab-pane">
        <div class="llm-config-panel">
            <div id="coding-delegation-bar" class="delegation-bar">
                <div class="delegation-bar-left">
                    <div class="delegation-bar-title">💻 Coding Delegation</div>
                    <div class="delegation-bar-desc">When enabled, the Main LLM will automatically delegate all coding, debugging, and code-review tasks to this model.</div>
                </div>
                <div class="toggle-wrap">
                    <span class="toggle-label off" id="coding-toggle-label">OFF</span>
                    <label class="toggle-switch">
                        <input type="checkbox" id="toggle-coding" onchange="toggleDelegation('coding',this.checked)">
                        <div class="toggle-track"></div>
                        <div class="toggle-thumb"></div>
                    </label>
                    <span class="toggle-label on" style="color:#6a9955">ON</span>
                </div>
            </div>

            <div class="llm-history-wrap">
                <label>⏱ Recent LLMs</label>
                <select class="llm-history-select" id="hist-coding" onchange="applyLLMHistory('coding',this.value)">
                    <option value="">— select a saved profile —</option>
                </select>
            </div>
            <div id="coding-llm-badge" class="llm-status-badge badge-warn"><span class="badge-dot"></span><span id="coding-llm-badge-text">Not configured</span></div>
            <div class="field-group">
                <label class="field-label" for="coding-llm-url">Base URL</label>
                <input type="text" id="coding-llm-url" class="field-input" placeholder="https://api.openai.com or http://localhost:1234">
                <div class="field-hint"><code>/v1/chat/completions</code> appended automatically.</div>
            </div>
            <div class="field-group">
                <label class="field-label" for="coding-llm-model">Model Name</label>
                <input type="text" id="coding-llm-model" class="field-input" placeholder="gpt-4o / deepseek-coder / codellama">
            </div>
            <div class="field-group">
                <label class="field-label" for="coding-llm-key">API Key</label>
                <input type="password" id="coding-llm-key" class="field-input" placeholder="sk-... (leave blank if not required)">
            </div>
            <div class="field-group">
                <label class="field-label" for="coding-llm-timeout">Request Timeout (seconds)</label>
                <input type="number" id="coding-llm-timeout" class="field-input" placeholder="300" min="10" max="3600">
                <div class="field-hint">Default 300s. Retries up to 2× on timeout.</div>
            </div>
            <hr class="divider">
            <button class="btn btn-green"  style="width:100%;padding:9px;margin-bottom:6px" onclick="saveCodingLLM()">💾 Save</button>
            <button class="btn btn-purple" style="width:100%;padding:9px" onclick="testCodingLLM()">⚡ Test Connection</button>
            <div id="coding-llm-status" class="llm-save-status"></div>
            <div id="coding-llm-test-result" class="test-result"></div>
        </div>
    </div>

    <!-- ===== MEDIA LLM ===== -->
    <div id="pane-media-llm" class="tab-pane">
        <div class="llm-config-panel">
            <div id="media-delegation-bar" class="delegation-bar">
                <div class="delegation-bar-left">
                    <div class="delegation-bar-title">🎬 Media Delegation</div>
                    <div class="delegation-bar-desc">When enabled, the Main LLM will automatically delegate all image generation, video creation, and media editing tasks to this model.</div>
                </div>
                <div class="toggle-wrap">
                    <span class="toggle-label off" id="media-toggle-label">OFF</span>
                    <label class="toggle-switch">
                        <input type="checkbox" id="toggle-media" onchange="toggleDelegation('media',this.checked)">
                        <div class="toggle-track"></div>
                        <div class="toggle-thumb"></div>
                    </label>
                    <span class="toggle-label on" style="color:#6a9955">ON</span>
                </div>
            </div>

            <div class="llm-history-wrap">
                <label>⏱ Recent LLMs</label>
                <select class="llm-history-select" id="hist-media" onchange="applyLLMHistory('media',this.value)">
                    <option value="">— select a saved profile —</option>
                </select>
            </div>
            <div id="media-llm-badge" class="llm-status-badge badge-warn"><span class="badge-dot"></span><span id="media-llm-badge-text">Not configured</span></div>
            <div style="background:#1a1a2a;border:1px solid #5a2d82;border-radius:4px;padding:8px 10px;margin-bottom:10px;font-size:0.75rem;color:#c586c0;flex-shrink:0;">
                🎬 <b>Media LLM</b> — for image generation, video creation &amp; editing. Use models like DALL-E, Stable Diffusion, or video generation APIs.
            </div>
            <div class="field-group">
                <label class="field-label" for="media-llm-url">Base URL</label>
                <input type="text" id="media-llm-url" class="field-input" placeholder="https://api.openai.com or http://localhost:7860">
                <div class="field-hint">Image/video model server. <code>/v1/chat/completions</code> appended automatically.</div>
            </div>
            <div class="field-group">
                <label class="field-label" for="media-llm-model">Model Name</label>
                <input type="text" id="media-llm-model" class="field-input" placeholder="dall-e-3 / stable-diffusion / runway-gen3">
            </div>
            <div class="field-group">
                <label class="field-label" for="media-llm-key">API Key</label>
                <input type="password" id="media-llm-key" class="field-input" placeholder="sk-... (leave blank if not required)">
            </div>
            <div class="field-group">
                <label class="field-label" for="media-llm-timeout">Request Timeout (seconds)</label>
                <input type="number" id="media-llm-timeout" class="field-input" placeholder="300" min="10" max="3600">
                <div class="field-hint">Image/video generation can be slow — increase for large models.</div>
            </div>
            <hr class="divider">
            <button class="btn btn-green"  style="width:100%;padding:9px;margin-bottom:6px" onclick="saveMediaLLM()">💾 Save</button>
            <button class="btn btn-purple" style="width:100%;padding:9px" onclick="testMediaLLM()">⚡ Test Connection</button>
            <div id="media-llm-status" class="llm-save-status"></div>
            <div id="media-llm-test-result" class="test-result"></div>
        </div>
    </div>

    <!-- ===== CONTENT LLM ===== -->
    <div id="pane-content-llm" class="tab-pane">
        <div class="llm-config-panel">
            <div id="content-delegation-bar" class="delegation-bar">
                <div class="delegation-bar-left">
                    <div class="delegation-bar-title">✍️ Content Delegation</div>
                    <div class="delegation-bar-desc">When enabled, the Main LLM will automatically delegate all blog posts, articles, SEO copy, newsletters, and web writing tasks to this model.</div>
                </div>
                <div class="toggle-wrap">
                    <span class="toggle-label off" id="content-toggle-label">OFF</span>
                    <label class="toggle-switch">
                        <input type="checkbox" id="toggle-content" onchange="toggleDelegation('content',this.checked)">
                        <div class="toggle-track"></div>
                        <div class="toggle-thumb"></div>
                    </label>
                    <span class="toggle-label on" style="color:#6a9955">ON</span>
                </div>
            </div>

            <div class="llm-history-wrap">
                <label>⏱ Recent LLMs</label>
                <select class="llm-history-select" id="hist-content" onchange="applyLLMHistory('content',this.value)">
                    <option value="">— select a saved profile —</option>
                </select>
            </div>
            <div id="content-llm-badge" class="llm-status-badge badge-warn"><span class="badge-dot"></span><span id="content-llm-badge-text">Not configured</span></div>
            <div style="background:#1a2a1a;border:1px solid #2d6e2d;border-radius:4px;padding:8px 10px;margin-bottom:10px;font-size:0.75rem;color:#6a9955;flex-shrink:0;">
                ✍️ <b>Content LLM</b> — for web content writing: blog posts, articles, SEO copy, social media text, newsletters, and marketing content.
            </div>
            <div class="field-group">
                <label class="field-label" for="content-llm-url">Base URL</label>
                <input type="text" id="content-llm-url" class="field-input" placeholder="https://api.openai.com or http://localhost:1234">
                <div class="field-hint">Web content writing model. <code>/v1/chat/completions</code> appended automatically.</div>
            </div>
            <div class="field-group">
                <label class="field-label" for="content-llm-model">Model Name</label>
                <input type="text" id="content-llm-model" class="field-input" placeholder="gpt-4o / claude-3 / mistral">
            </div>
            <div class="field-group">
                <label class="field-label" for="content-llm-key">API Key</label>
                <input type="password" id="content-llm-key" class="field-input" placeholder="sk-... (leave blank if not required)">
            </div>
            <div class="field-group">
                <label class="field-label" for="content-llm-timeout">Request Timeout (seconds)</label>
                <input type="number" id="content-llm-timeout" class="field-input" placeholder="300" min="10" max="3600">
                <div class="field-hint">Default 300s. Increase for long-form content generation.</div>
            </div>
            <hr class="divider">
            <button class="btn btn-green" style="width:100%;padding:9px;margin-bottom:6px" onclick="saveContentLLM()">💾 Save</button>
            <button class="btn btn-teal"  style="width:100%;padding:9px" onclick="testContentLLM()">⚡ Test Connection</button>
            <div id="content-llm-status" class="llm-save-status"></div>
            <div id="content-llm-test-result" class="test-result"></div>
        </div>
    </div>


    <!-- ===== WEB SEARCH ===== -->
    <div id="pane-websearch" class="tab-pane">
        <div style="max-width:700px;display:flex;flex-direction:column;gap:14px;overflow-y:auto;flex:1;">
            <div>
                <div class="field-label" style="margin-bottom:8px;">Search Provider</div>
                <div class="search-provider-grid">
                    <div class="provider-card selected" id="prov-duckduckgo" onclick="selectProvider('duckduckgo')">
                        <div class="provider-icon">🦆</div>
                        <div class="provider-name">DuckDuckGo</div>
                        <div style="font-size:0.65rem;color:#6a9955;margin-top:3px;">No key needed</div>
                    </div>
                    <div class="provider-card" id="prov-google" onclick="selectProvider('google')">
                        <div class="provider-icon">🔍</div>
                        <div class="provider-name">Google</div>
                        <div style="font-size:0.65rem;color:#ce9178;margin-top:3px;">API key required</div>
                    </div>
                    <div class="provider-card" id="prov-brave" onclick="selectProvider('brave')">
                        <div class="provider-icon">🦁</div>
                        <div class="provider-name">Brave Search</div>
                        <div style="font-size:0.65rem;color:#ce9178;margin-top:3px;">API key required</div>
                    </div>
                </div>
            </div>

            <div id="search-key-fields" style="display:none;">
                <div class="field-group">
                    <label class="field-label" for="search-api-key">API Key</label>
                    <input type="password" id="search-api-key" class="field-input" placeholder="Paste your API key...">
                    <div class="field-hint" id="search-key-hint">Required for the selected provider.</div>
                </div>
                <div class="field-group" id="search-cx-group" style="display:none;">
                    <label class="field-label" for="search-cx">Custom Search Engine ID (cx)</label>
                    <input type="text" id="search-cx" class="field-input" placeholder="Google Custom Search Engine ID">
                    <div class="field-hint">Create one at <code>programmablesearchengine.google.com</code></div>
                </div>
            </div>

            <div style="display:flex;gap:8px;">
                <button class="btn btn-green" onclick="saveSearchConfig()" style="padding:8px 18px;">💾 Save</button>
                <button class="btn btn-teal"  onclick="testSearch()"       style="padding:8px 18px;">⚡ Test Search</button>
            </div>
            <div id="search-status" class="llm-save-status"></div>

            <div>
                <div class="field-label" style="margin-bottom:6px;">Test Query</div>
                <div style="display:flex;gap:8px;">
                    <input type="text" id="search-test-query" placeholder="e.g. latest AI news" style="flex:1;" onkeypress="if(event.key==='Enter')testSearch()">
                </div>
            </div>
            <div class="search-test-box" id="search-test-box"></div>

            <div style="background:#1e2a1e;border:1px solid #2d6e2d;border-radius:6px;padding:12px;">
                <div style="font-size:0.75rem;font-weight:bold;color:#6a9955;margin-bottom:6px;">🤖 Autonomous Usage</div>
                <div style="font-size:0.75rem;color:#666;line-height:1.6;">
                    The agent can call <code>web_search(query)</code> and <code>fetch_url(url)</code> autonomously at any time. Use web_search to find current information, news, research, or facts. Use fetch_url to read full article content after searching.
                </div>
            </div>
        </div>
    </div>


    <!-- ===== SOCIAL ===== -->
    <div id="pane-social" class="tab-pane">
        <div style="display:flex;flex-direction:column;gap:0;flex:1;overflow:hidden;">
            <div style="display:flex;gap:8px;margin-bottom:10px;flex-shrink:0;align-items:center;">
                <button class="btn btn-green" onclick="saveSocialConfig()" style="padding:7px 16px;">💾 Save All</button>
                <div id="social-save-status" style="font-size:0.75rem;color:#6a9955;min-height:1em;"></div>
            </div>
            <div class="social-grid">

                <!-- Discord -->
                <div class="social-card">
                    <div class="social-card-hdr">
                        <span class="social-icon">💬</span>
                        <span class="social-name">Discord</span>
                        <span class="social-status-dot" id="dot-discord"></span>
                    </div>
                    <div class="social-field"><label>Bot Token</label><input type="password" id="soc-discord-token" placeholder="Bot token..."></div>
                    <div class="social-field"><label>Channel ID</label><input type="text" id="soc-discord-channel" placeholder="Channel ID..."></div>
                    <div class="social-btns">
                        <button class="btn btn-teal" style="flex:1;padding:5px;" onclick="testSocial('discord')">⚡ Test</button>
                    </div>
                    <div class="social-save-status" id="soc-status-discord"></div>
                </div>

                <!-- Twitter/X -->
                <div class="social-card">
                    <div class="social-card-hdr">
                        <span class="social-icon">🐦</span>
                        <span class="social-name">Twitter / X</span>
                        <span class="social-status-dot" id="dot-twitter"></span>
                    </div>
                    <div class="social-field"><label>API Key (Consumer Key)</label><input type="password" id="soc-twitter-key" placeholder="API Key..."></div>
                    <div class="social-field"><label>API Secret</label><input type="password" id="soc-twitter-secret" placeholder="API Secret..."></div>
                    <div class="social-field"><label>Access Token</label><input type="password" id="soc-twitter-token" placeholder="Access Token..."></div>
                    <div class="social-field"><label>Access Token Secret</label><input type="password" id="soc-twitter-token-sec" placeholder="Access Token Secret..."></div>
                    <div class="social-btns">
                        <button class="btn btn-teal" style="flex:1;padding:5px;" onclick="testSocial('twitter')">⚡ Test</button>
                    </div>
                    <div class="social-save-status" id="soc-status-twitter"></div>
                </div>

                <!-- Telegram -->
                <div class="social-card">
                    <div class="social-card-hdr">
                        <span class="social-icon">✈️</span>
                        <span class="social-name">Telegram</span>
                        <span class="social-status-dot" id="dot-telegram"></span>
                    </div>
                    <div class="social-field"><label>Bot Token</label><input type="password" id="soc-telegram-token" placeholder="Bot token from @BotFather..."></div>
                    <div class="social-field"><label>Chat ID</label><input type="text" id="soc-telegram-chat" placeholder="Chat or group ID..."></div>
                    <div class="social-btns">
                        <button class="btn btn-teal" style="flex:1;padding:5px;" onclick="testSocial('telegram')">⚡ Test</button>
                    </div>
                    <div class="social-save-status" id="soc-status-telegram"></div>
                </div>

                <!-- Slack -->
                <div class="social-card">
                    <div class="social-card-hdr">
                        <span class="social-icon">🔷</span>
                        <span class="social-name">Slack</span>
                        <span class="social-status-dot" id="dot-slack"></span>
                    </div>
                    <div class="social-field"><label>Bot Token (xoxb-...)</label><input type="password" id="soc-slack-token" placeholder="xoxb-..."></div>
                    <div class="social-field"><label>Channel</label><input type="text" id="soc-slack-channel" placeholder="#general or channel ID..."></div>
                    <div class="social-btns">
                        <button class="btn btn-teal" style="flex:1;padding:5px;" onclick="testSocial('slack')">⚡ Test</button>
                    </div>
                    <div class="social-save-status" id="soc-status-slack"></div>
                </div>

                <!-- WhatsApp -->
                <div class="social-card">
                    <div class="social-card-hdr">
                        <span class="social-icon">📱</span>
                        <span class="social-name">WhatsApp (Twilio)</span>
                        <span class="social-status-dot" id="dot-whatsapp"></span>
                    </div>
                    <div class="social-field"><label>Twilio Account SID</label><input type="password" id="soc-wa-sid" placeholder="ACxxxxxxxx..."></div>
                    <div class="social-field"><label>Auth Token</label><input type="password" id="soc-wa-token" placeholder="Auth token..."></div>
                    <div class="social-field"><label>From Number (+1234...)</label><input type="text" id="soc-wa-from" placeholder="+14155238886"></div>
                    <div class="social-field"><label>To Number (+1234...)</label><input type="text" id="soc-wa-to" placeholder="+15551234567"></div>
                    <div class="social-btns">
                        <button class="btn btn-teal" style="flex:1;padding:5px;" onclick="testSocial('whatsapp')">⚡ Test</button>
                    </div>
                    <div class="social-save-status" id="soc-status-whatsapp"></div>
                </div>

            </div>
        </div>
    </div>

    <!-- ===== MODULES ===== -->
    <div id="pane-modules" class="tab-pane">
        <div class="mod-tabs">
            <div class="mod-tab active" onclick="switchModTab('browse',this)">Browse</div>
            <div class="mod-tab"        onclick="switchModTab('installed',this)">Installed</div>
            <div class="mod-tab"        onclick="switchModTab('gomod',this)">go.mod</div>
        </div>
        <div id="mod-pane-browse" style="display:flex;flex-direction:column;flex:1;overflow:hidden;">
            <div class="mod-search-row">
                <input type="text" id="mod-search-input" placeholder="Search pkg.go.dev..." onkeypress="if(event.key==='Enter')searchModules()">
                <button class="btn btn-blue" onclick="searchModules()" style="padding:6px 14px;">Search</button>
            </div>
            <div class="mod-status" id="mod-search-status"></div>
            <div class="scroll-list" id="mod-results"><div class="empty-state">Search for Go packages above.</div></div>
        </div>
        <div id="mod-pane-installed" style="display:none;flex-direction:column;flex:1;overflow:hidden;">
            <div style="display:flex;gap:6px;margin-bottom:8px;flex-shrink:0;max-width:500px;">
                <input type="text" id="mod-manual-input" placeholder="Module path (e.g. github.com/user/pkg)" style="flex:1;" onkeypress="if(event.key==='Enter')addModuleManual()">
                <button class="btn btn-green" onclick="addModuleManual()" style="padding:6px 12px;">+ Add</button>
            </div>
            <div class="mod-status" id="mod-install-status"></div>
            <div class="scroll-list" id="mod-installed-list"><div class="empty-state">Loading...</div></div>
        </div>
        <div id="mod-pane-gomod" style="display:none;flex-direction:column;flex:1;overflow:hidden;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;flex-shrink:0;">
                <span style="font-size:0.78rem;color:#aaa;">go.mod contents</span>
                <button class="btn btn-teal" onclick="loadGoMod()" style="padding:4px 10px;font-size:0.72rem;">↻ Refresh</button>
            </div>
            <textarea id="mod-gomod-content" style="flex:1;resize:none;font-family:monospace;font-size:0.76rem;line-height:1.5;padding:8px;background:#1a1a1a;border:1px solid #333;border-radius:4px;color:#d4d4d4;" readonly placeholder="go.mod not found"></textarea>
        </div>
    </div>

    <!-- ===== HOST ===== -->
    <div id="pane-host" class="tab-pane">
        <div class="host-panel">
            <div id="host-badge" class="llm-status-badge badge-warn"><span class="badge-dot"></span><span id="host-badge-text">No credentials set</span></div>
            <div class="warning-box"><b>⚠ Full Root Access</b><br>When credentials are saved, every shell command the agent runs will be executed via <code>sudo</code> as the specified user. The agent will have unrestricted host access.</div>
            <div class="field-group">
                <label class="field-label" for="host-user">Username</label>
                <input type="text" id="host-user" class="field-input" placeholder="root (or another sudoer)">
                <div class="field-hint">Leave blank to default to <code>root</code>.</div>
            </div>
            <div class="field-group">
                <label class="field-label" for="host-pass">Password</label>
                <input type="password" id="host-pass" class="field-input" placeholder="sudo / root password">
                <div class="field-hint">Passed to <code>sudo -S</code>. Stored in <code>config.json</code>.</div>
            </div>
            <hr class="divider">
            <button class="btn btn-green" style="width:100%;padding:9px;margin-bottom:6px" onclick="saveHostConfig()">💾 Save Credentials</button>
            <button class="btn btn-red"   style="width:100%;padding:9px;margin-bottom:6px" onclick="clearHostConfig()">🗑 Clear Credentials</button>
            <button class="btn btn-teal"  style="width:100%;padding:9px" onclick="verifyHostAccess()">⚡ Verify Root Access</button>
            <div id="host-status" class="llm-save-status"></div>
            <div id="host-verify-result" class="verify-result"></div>
        </div>
    </div>

    <!-- ===== NEURAL NETWORKS ===== -->
    <div id="pane-neural" class="tab-pane">

        <!-- Header banner -->
        <div style="background:linear-gradient(135deg,#1a0a2a,#0a1a2a);border:1px solid #5a2d82;border-radius:8px;padding:10px 16px;margin-bottom:12px;font-size:0.78rem;">
            <span style="color:#c586c0;font-weight:bold;">🧠 Expert NN Chain</span>
            <span style="color:#888;margin-left:8px;">NN1 always on · Experts wake on demand · Neural Scaling (1/2/4 layers) · Superposition encoding</span>
        </div>

        <!-- Sub-nav -->
        <div style="display:flex;gap:6px;margin-bottom:12px;flex-wrap:wrap;" id="nn-subnav">
            <button class="btn btn-purple" id="ntab-btn-chain"   onclick="showNTab('chain')"   style="padding:5px 14px;font-size:0.75rem;">⛓ Expert Chain</button>
            <button class="btn"           id="ntab-btn-basic"   onclick="showNTab('basic')"   style="padding:5px 14px;font-size:0.75rem;background:#1e1e2e;border:1px solid #3a3a5a;color:#aaa;">🗄 Base NNs</button>
            <button class="btn"           id="ntab-btn-routing" onclick="showNTab('routing')" style="padding:5px 14px;font-size:0.75rem;background:#1e1e2e;border:1px solid #3a3a5a;color:#aaa;">🗺 Routing</button>
            <button class="btn"           id="ntab-btn-train"   onclick="showNTab('train')"   style="padding:5px 14px;font-size:0.75rem;background:#1e1e2e;border:1px solid #3a3a5a;color:#aaa;">⚙ Training</button>
        </div>

        <!-- ── Expert Chain tab ── -->
        <div id="nn-tab-chain">
            <div style="display:flex;gap:8px;margin-bottom:10px;align-items:center;">
                <button class="btn btn-teal"  onclick="loadChain()" style="padding:6px 14px;">↻ Refresh</button>
                <button class="btn btn-green" onclick="document.getElementById('spawn-expert-form').style.display='block'" style="padding:6px 14px;">＋ Spawn Expert</button>
                <span id="chain-summary" style="font-size:0.73rem;color:#888;"></span>
            </div>
            <div id="nn-chain-grid" style="display:grid;grid-template-columns:repeat(auto-fill,minmax(270px,1fr));gap:10px;margin-bottom:12px;"></div>
            <!-- Spawn form -->
            <div id="spawn-expert-form" style="display:none;background:#1a0a2a;border:1px solid #5a2d82;border-radius:8px;padding:14px;">
                <div style="color:#c586c0;font-weight:bold;margin-bottom:10px;">＋ Spawn New Expert NN</div>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:8px;">
                    <div><div class="field-label" style="margin-bottom:3px;">Name</div>
                    <input type="text" id="exp-name"   class="field-input" placeholder="e.g. Marketing Expert"></div>
                    <div><div class="field-label" style="margin-bottom:3px;">Domain</div>
                    <input type="text" id="exp-domain" class="field-input" placeholder="e.g. marketing"></div>
                </div>
                <div class="field-label" style="margin-bottom:3px;">Skills (comma-separated)</div>
                <input type="text" id="exp-skills" class="field-input" style="width:100%;margin-bottom:8px;" placeholder="e.g. copywriting, brand strategy, analytics">
                <div style="display:flex;gap:8px;align-items:center;margin-top:4px;">
                    <div><div class="field-label" style="margin-bottom:3px;">Scale</div>
                    <select id="exp-scale" class="field-input" style="width:130px;">
                        <option value="1">1 — Fast</option>
                        <option value="2" selected>2 — Standard</option>
                        <option value="4">4 — Deep Expert</option>
                    </select></div>
                    <button class="btn btn-green" onclick="spawnExpert()" style="padding:8px 18px;margin-top:18px;">Spawn</button>
                    <button class="btn" onclick="document.getElementById('spawn-expert-form').style.display='none'" style="padding:8px 12px;margin-top:18px;background:#2d2d2d;border:1px solid #555;">Cancel</button>
                </div>
                <div id="spawn-status" class="llm-save-status" style="margin-top:8px;"></div>
            </div>
        </div>

        <!-- ── Base NNs tab ── -->
        <div id="nn-tab-basic" style="display:none;">
            <div style="display:flex;gap:8px;margin-bottom:10px;flex-wrap:wrap;align-items:flex-end;">
                <div style="flex:1;min-width:200px;">
                    <input type="text"   id="nn-spawn-name" class="field-input" placeholder="network_name" style="margin-bottom:6px;">
                    <input type="number" id="nn-spawn-cap"  class="field-input" placeholder="Capacity (default 500)">
                </div>
                <button class="btn btn-green" onclick="nnSpawn()"      style="padding:8px 16px;">⚡ Spawn</button>
                <button class="btn btn-teal"  onclick="loadNeuralNets()" style="padding:8px 16px;">↻ Refresh</button>
            </div>
            <div id="nn-spawn-status" class="llm-save-status" style="margin-bottom:10px;"></div>
            <div id="nn-nets-list" style="margin-bottom:14px;"></div>
            <div style="background:#1a1a2a;border:1px solid #3a3a5a;border-radius:6px;padding:12px;margin-bottom:10px;">
                <div class="field-label" style="margin-bottom:8px;">🔍 Test Recall</div>
                <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:flex-end;">
                    <input type="text"   id="nn-recall-net"   class="field-input" placeholder="network (blank=agent_memory)" style="flex:1;min-width:120px;">
                    <input type="text"   id="nn-recall-query" class="field-input" placeholder="query" style="flex:2;min-width:160px;">
                    <input type="number" id="nn-recall-k"     class="field-input" value="3" style="width:60px;">
                    <button class="btn btn-purple" onclick="nnTestRecall()" style="padding:7px 14px;">Recall</button>
                </div>
                <div id="nn-recall-result" style="margin-top:10px;font-size:0.77rem;color:#9cdcfe;white-space:pre-wrap;max-height:200px;overflow-y:auto;"></div>
            </div>
            <div style="background:#1a1a2a;border:1px solid #3a3a5a;border-radius:6px;padding:12px;">
                <div class="field-label" style="margin-bottom:8px;">💾 Manual Store</div>
                <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:flex-end;">
                    <input type="text" id="nn-store-net" class="field-input" placeholder="network" style="flex:1;min-width:120px;">
                    <input type="text" id="nn-store-key" class="field-input" placeholder="key"   style="flex:2;">
                    <input type="text" id="nn-store-val" class="field-input" placeholder="value" style="flex:2;">
                    <button class="btn btn-green" onclick="nnManualStore()" style="padding:7px 14px;">Store</button>
                </div>
                <div id="nn-store-result" style="margin-top:8px;font-size:0.77rem;color:#6a9955;"></div>
            </div>
        </div>

        <!-- ── Routing Table tab ── -->
        <div id="nn-tab-routing" style="display:none;">
            <div style="display:flex;gap:8px;margin-bottom:10px;align-items:center;">
                <input type="text" id="routing-search" class="field-input" placeholder="Filter by task, domain, or action..." oninput="filterRouting()" style="flex:1;max-width:420px;">
                <button class="btn btn-teal" onclick="loadRouting()" style="padding:6px 14px;">↻ Refresh</button>
                <span id="routing-count" style="font-size:0.73rem;color:#888;align-self:center;"></span>
            </div>
            <div id="routing-list" style="max-height:420px;overflow-y:auto;font-size:0.74rem;"></div>
        </div>

        <!-- ── Training tab ── -->
        <div id="nn-tab-train" style="display:none;">
            <div style="background:#1a1a2a;border:1px solid #3a3a5a;border-radius:6px;padding:12px;margin-bottom:12px;">
                <div class="field-label" style="margin-bottom:8px;">Train an Expert NN</div>
                <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:flex-end;">
                    <input type="text" id="train-domain" class="field-input" placeholder="domain  e.g. finance" style="flex:1;min-width:150px;">
                    <select id="train-mode" class="field-input" style="width:150px;">
                        <option value="self">Self-train (fast)</option>
                        <option value="llm">LLM-train (quality)</option>
                    </select>
                    <button class="btn btn-purple" onclick="startTraining()" style="padding:7px 18px;">▶ Train</button>
                </div>
                <div id="train-status" class="llm-save-status" style="margin-top:8px;"></div>
            </div>
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
                <span class="field-label">Training Jobs</span>
                <button class="btn btn-teal" onclick="loadJobs()" style="padding:4px 12px;font-size:0.72rem;">↻ Refresh</button>
            </div>
            <div id="training-jobs-list" style="max-height:320px;overflow-y:auto;"></div>
        </div>

    </div>

    <!-- ===== AUDIT ===== -->
    <div id="pane-audit" class="tab-pane">
        <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:10px;">
            <input type="text" class="search-bar" id="audit-search" placeholder="Search audit log..." oninput="filterAudit()" style="flex:1;min-width:200px;max-width:420px;">
            <select id="audit-level-filter" class="field-input" style="width:130px;padding:6px;" onchange="filterAudit()">
                <option value="">All levels</option>
                <option value="INFO">INFO</option>
                <option value="ERROR">ERROR</option>
                <option value="SUCCESS">SUCCESS</option>
                <option value="ACTION">ACTION</option>
                <option value="CONTEXT">CONTEXT</option>
                <option value="NEURAL">NEURAL</option>
                <option value="MEMORY">MEMORY</option>
                <option value="CODING_LLM">CODING_LLM</option>
                <option value="MEDIA_LLM">MEDIA_LLM</option>
                <option value="CONTENT_LLM">CONTENT_LLM</option>
                <option value="CONFIG">CONFIG</option>
                <option value="TASK">TASK</option>
                <option value="MODULE">MODULE</option>
            </select>
            <button class="btn btn-teal"  onclick="printAuditLog()" style="padding:6px 14px;">🖨 Print</button>
            <button class="btn btn-green" onclick="exportAuditCSV()" style="padding:6px 14px;">⬇ Export CSV</button>
            <span id="audit-count" style="font-size:0.75rem;color:#888;align-self:center;"></span>
        </div>
        <div class="scroll-list" id="audit-list"><div class="empty-state">No log entries yet.</div></div>
    </div>

</div><!-- end tab-content -->

<!-- Session modal -->
<div class="modal-overlay" id="session-modal" onclick="closeModal(event)">
    <div class="modal">
        <div class="modal-hdr">
            <span id="modal-title"></span>
            <button class="close-btn" onclick="document.getElementById('session-modal').classList.remove('open')">Close</button>
        </div>
        <div class="modal-body" id="modal-body"></div>
    </div>
</div>

<script>
var logs=[], memories=[], sessions=[];
var tabNames=['chat','tasks','sessions','personality','main-llm','coding-llm','media-llm','content-llm','websearch','social','modules','host','neural','audit'];

function switchTab(name){
    document.querySelectorAll('.ttab').forEach(function(t,i){ t.classList.toggle('active', tabNames[i]===name); });
    tabNames.forEach(function(n){ var p=document.getElementById('pane-'+n); if(p) p.classList.toggle('active', n===name); });
}

/* ---- Data polling ---- */
async function fetchData(){
    fetchTasks();
    fetchChatMessages();
    try{ var d=await(await fetch('/api/logs')).json(); if(d&&d.length!==logs.length){logs=d;renderActivity();filterAudit(); var wasRunning=(logs.filter(function(l){return l.level==='INFO';}).length>0); showTyping(false);} }catch(e){}
    try{ var d=await(await fetch('/api/memory')).json(); if(d&&d.length!==memories.length){memories=d;filterMemory();} }catch(e){}
    try{ var d=await(await fetch('/api/sessions')).json(); if(d&&d.length!==sessions.length){sessions=d;renderSessions();} }catch(e){}
}

/* ---- Chat ---- */
var chatMsgs=[], activityVisible=true;

function toggleActivity(){
    var panel=document.getElementById('chat-activity-panel');
    activityVisible=!activityVisible;
    panel.style.display=activityVisible?'flex':'none';
    document.querySelector('.activity-toggle').textContent=activityVisible?'hide':'show';
}

async function fetchChatMessages(){
    try{
        var d=await(await fetch('/api/chat_messages')).json();
        if(JSON.stringify(d)!==JSON.stringify(chatMsgs)){
            chatMsgs=d||[];
            renderChatMessages();
        }
    }catch(e){}
}

function renderChatMessages(){
    var c=document.getElementById('chat-messages');
    var empty=document.getElementById('chat-empty');
    if(!chatMsgs||!chatMsgs.length){
        if(empty) empty.style.display='block';
        c.innerHTML='<div class="empty-state" id="chat-empty" style="margin:auto;">Send a message to get started.</div>';
        return;
    }
    var h='';
    chatMsgs.forEach(function(m){
        var isUser=m.role==='user';
        var t=new Date(m.timestamp).toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'});
        h+='<div class="msg-row '+(isUser?'user':'agent')+'">'
          +'<div class="msg-avatar '+(isUser?'user-av':'agent-av')+'">'+(isUser?'👤':'🤖')+'</div>'
          +'<div>'
          +'<div class="msg-bubble">'+esc(m.content)+'</div>'
          +'<div class="msg-time">'+t+'</div>'
          +'</div></div>';
    });
    c.innerHTML=h;
    c.scrollTop=c.scrollHeight;
}

function renderActivity(){
    var c=document.getElementById('activity-feed');
    if(!logs||!logs.length){c.innerHTML='';return;}
    var h='';
    logs.slice(-80).forEach(function(l){
        var t=new Date(l.timestamp).toLocaleTimeString([],{hour:'2-digit',minute:'2-digit',second:'2-digit'});
        h+='<div class="act-entry act-'+l.level+'">'+t+' '+esc(l.content)+'</div>';
    });
    c.innerHTML=h; c.scrollTop=c.scrollHeight;
}

function showTyping(on){
    var el=document.getElementById('agent-typing');
    if(el) el.style.display=on?'inline':'none';
}

/* ---- Memory ---- */
function filterMemory(){
    var q=document.getElementById('mem-search').value.toLowerCase();
    var c=document.getElementById('memory-list');
    var f=memories.filter(function(m){ return m.content.toLowerCase().includes(q)||((m.tags||[]).join(' ').toLowerCase().includes(q)); });
    if(!f.length){c.innerHTML='<div class="empty-state">No memories match.</div>';return;}
    var h='';
    f.slice().reverse().forEach(function(m){
        var tags=(m.tags||[]).map(function(t){return'<span class="tag">'+esc(t)+'</span>';}).join(' ');
        h+='<div class="mem-item"><div class="mem-meta"><div>'+tags+'</div><div>'+new Date(m.created_at).toLocaleDateString()+'</div></div>'+esc(m.content)+'</div>';
    });
    c.innerHTML=h;
}

/* ---- Sessions ---- */
function renderSessions(){
    var c=document.getElementById('session-list');
    if(!sessions.length){c.innerHTML='<div class="empty-state">No sessions yet.</div>';return;}
    var h='';
    sessions.slice().reverse().forEach(function(s){
        h+='<div class="session-item" onclick="viewSession('+s.id+')">'
          +'<div class="session-title">#'+s.id+': '+esc(s.title)+'</div>'
          +'<div class="session-meta">'+new Date(s.created_at).toLocaleString()+' &bull; '+(s.logs||[]).length+' entries</div></div>';
    });
    c.innerHTML=h;
}
function viewSession(id){
    var s=sessions.find(function(x){return x.id===id;});
    if(!s) return;
    document.getElementById('modal-title').textContent='#'+s.id+': '+s.title;
    var h='';
    (s.logs||[]).forEach(function(l){
        h+='<div class="log-entry log-'+l.level+'">['+new Date(l.timestamp).toLocaleTimeString()+'] <b>'+l.level+'</b>: '+esc(l.content)+'</div>';
    });
    document.getElementById('modal-body').innerHTML=h||'<div class="empty-state">No entries.</div>';
    document.getElementById('session-modal').classList.add('open');
}
function closeModal(e){ if(e.target.id==='session-modal') document.getElementById('session-modal').classList.remove('open'); }

/* ---- Audit ---- */

/* ─── Expert NN Chain JS ────────────────────────────────────────────── */
var nnChainData = [];
var routingData = [];

function showNTab(tab) {
    ['chain','basic','routing','train'].forEach(function(t) {
        var el = document.getElementById('nn-tab-'+t);
        var btn = document.getElementById('ntab-btn-'+t);
        if (el) el.style.display = t===tab ? '' : 'none';
        if (btn) {
            btn.style.background = t===tab ? '' : '#1e1e2e';
            btn.style.borderColor = t===tab ? '' : '#3a3a5a';
            btn.style.color = t===tab ? '' : '#aaa';
            btn.classList.toggle('btn-purple', t===tab);
        }
    });
    if (tab==='chain')   loadChain();
    if (tab==='routing') loadRouting();
    if (tab==='train')   loadJobs();
    if (tab==='basic')   loadNeuralNets();
}

async function loadChain() {
    try {
        var data = await (await fetch('/api/nn/chain')).json();
        nnChainData = data || [];
        var grid = document.getElementById('nn-chain-grid');
        var sumEl = document.getElementById('chain-summary');
        if (!data || !data.length) {
            grid.innerHTML = '<div class="empty-state">No chain data yet — agent not started.</div>';
            return;
        }
        var nn1 = data.find(function(d){return d.is_nn1;});
        var experts = data.filter(function(d){return !d.is_nn1;});
        if (sumEl) sumEl.textContent = 'NN1 routes: '+(nn1?nn1.route_count||0:0)+' · Experts: '+experts.length;
        var html = '';
        // NN1 card first
        if (nn1) {
            html += '<div style="background:linear-gradient(135deg,#1a0a2a,#0a1028);border:2px solid #5a2d82;border-radius:8px;padding:12px;">'
                  + '<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">'
                  + '<span style="font-size:1.2rem;">🧠</span>'
                  + '<span style="font-weight:bold;color:#c586c0;">'+esc(nn1.name)+'</span>'
                  + '<span style="background:#5a2d82;color:#e0e0ff;font-size:0.65rem;padding:2px 6px;border-radius:10px;margin-left:auto;">ALWAYS AWAKE</span>'
                  + '</div>'
                  + '<div style="font-size:0.72rem;color:#888;display:grid;grid-template-columns:1fr 1fr;gap:2px;">'
                  + '<span>Scale: '+nn1.scale_level+' layers</span>'
                  + '<span>Trained: '+nn1.train_count+'</span>'
                  + '<span>Routes: '+(nn1.route_count||0)+'</span>'
                  + '<span>Wakes: '+nn1.wake_count+'</span>'
                  + '</div></div>';
        }
        // Expert cards
        experts.forEach(function(ex) {
            var awake = ex.status === 'awake' || ex.status === 'training';
            var training = ex.status === 'training';
            var inRam = ex.in_ram === true;
            var statusColor = training ? '#dcdcaa' : inRam ? '#6a9955' : '#3a3a5a';
            var statusText  = training ? '⚙ training' : inRam ? '⚡ awake (RAM)' : '💤 dormant';
            var statusBg    = training ? '#2a2a1a' : inRam ? '#1a2a1a' : '#1a1a2a';
            var scaleBar = '';
            for (var s=0;s<4;s++) scaleBar += '<div style="width:6px;height:14px;background:'+(s<ex.scale_level?'#9b6ecf':'#2d2d2d')+';border-radius:2px;"></div>';
            html += '<div style="background:'+statusBg+';border:1px solid '+(inRam?'#3a5a3a':'#3a3a5a')+';border-radius:8px;padding:12px;position:relative;">'
                  + (training ? '<div style="position:absolute;top:6px;right:6px;width:8px;height:8px;border-radius:50%;background:#dcdcaa;animation:pulse 1s infinite;"></div>' : '')
                  + '<div style="display:flex;align-items:flex-start;gap:8px;margin-bottom:6px;">'
                  + '<div style="flex:1;"><div style="font-weight:bold;color:#9cdcfe;font-size:0.85rem;">'+esc(ex.name)+'</div>'
                  + '<div style="font-size:0.68rem;color:#666;margin-top:1px;">domain: '+esc(ex.domain)+'</div></div>'
                  + '<div style="display:flex;gap:2px;align-items:center;">'+scaleBar+'</div>'
                  + '</div>'
                  + '<div style="font-size:0.7rem;color:#888;margin-bottom:6px;">'+(ex.skills||[]).slice(0,3).map(esc).join(' · ')+'</div>'
                  + '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">'
                  + '<span style="font-size:0.7rem;color:#666;">trained: '+ex.train_count+' · wakes: '+ex.wake_count+(inRam?' · <span style="color:#6a9955;">in RAM</span>':'')+'</span>'
                  + '<span style="font-size:0.68rem;color:'+statusColor+';">'+statusText+'</span>'
                  + '</div>'
                  + '<div style="display:flex;gap:6px;">'
                  + '<button class="btn" onclick="quickTrain(\''+esc(ex.domain)+'\',\'self\')" style="flex:1;padding:4px;font-size:0.68rem;background:#1e2e1e;border:1px solid #3a5a3a;color:#6a9955;">⚡ Self-train</button>'
                  + '<button class="btn" onclick="quickTrain(\''+esc(ex.domain)+'\',\'llm\')"  style="flex:1;padding:4px;font-size:0.68rem;background:#1e1e2e;border:1px solid #3a3a5a;color:#9cdcfe;">🤖 LLM-train</button>'
                  + '</div></div>';
        });
        grid.innerHTML = html;
    } catch(e) {
        document.getElementById('nn-chain-grid').innerHTML = '<div style="color:#f44747;">Error: '+e+'</div>';
    }
}

async function spawnExpert() {
    var name   = document.getElementById('exp-name').value.trim();
    var domain = document.getElementById('exp-domain').value.trim();
    var skills = document.getElementById('exp-skills').value.trim();
    var scale  = parseInt(document.getElementById('exp-scale').value);
    if (!name || !domain) { showStatus('spawn-status','Name and domain required.',false); return; }
    try {
        var d = await (await fetch('/api/nn/chain/spawn',{
            method:'POST', headers:{'Content-Type':'application/json'},
            body: JSON.stringify({name,domain,skills,scale})
        })).json();
        showStatus('spawn-status', '✓ '+d.result, true);
        setTimeout(loadChain, 800);
    } catch(e) { showStatus('spawn-status','✗ '+e, false); }
}

async function quickTrain(domain, mode) {
    try {
        var d = await (await fetch('/api/nn/chain/train',{
            method:'POST', headers:{'Content-Type':'application/json'},
            body: JSON.stringify({Domain:domain, Mode:mode})
        })).json();
        showStatus('spawn-status', '✓ '+d.result, true);
        setTimeout(loadChain, 1200);
    } catch(e) { alert('Train error: '+e); }
}

async function loadRouting() {
    try {
        var data = await (await fetch('/api/nn/chain/routing')).json();
        routingData = data || [];
        renderRouting();
    } catch(e) { document.getElementById('routing-list').innerHTML = '<div style="color:#f44747;">Error: '+e+'</div>'; }
}

function filterRouting() {
    renderRouting();
}

function renderRouting() {
    var q = (document.getElementById('routing-search')||{value:''}).value.toLowerCase();
    var filtered = routingData.filter(function(r) {
        return !q || r.task_pattern.toLowerCase().includes(q)
                  || r.domain.toLowerCase().includes(q)
                  || r.action.toLowerCase().includes(q);
    });
    var el = document.getElementById('routing-list');
    var cnt = document.getElementById('routing-count');
    if (cnt) cnt.textContent = filtered.length+' / '+routingData.length;
    if (!filtered.length) { el.innerHTML='<div class="empty-state">No routing entries yet.</div>'; return; }
    var domainColors = {coding:'#9cdcfe',content:'#dcdcaa',media:'#c586c0',finance:'#6a9955',
                        legal:'#4ec9b0',medical:'#f44747',data:'#ce9178',science:'#569cd6',
                        search:'#888',system:'#aaa',memory:'#c586c0',tasks:'#dcdcaa',comms:'#6a9955'};
    var html = filtered.slice().reverse().map(function(r) {
        var col = domainColors[r.domain] || '#888';
        return '<div style="display:flex;align-items:center;gap:8px;padding:5px 8px;border-bottom:1px solid #2a2a3a;">'
             + '<span style="color:'+col+';min-width:70px;font-size:0.72rem;">'+esc(r.domain)+'</span>'
             + '<span style="color:#888;min-width:110px;font-size:0.7rem;">→ '+esc(r.action)+'</span>'
             + '<span style="color:#ccc;flex:1;">'+esc(truncStr(r.task_pattern,60))+'</span>'
             + '<span style="color:#555;font-size:0.66rem;">×'+r.use_count+'</span>'
             + '</div>';
    }).join('');
    el.innerHTML = html;
}

function truncStr(s, n) { return s && s.length > n ? s.slice(0, n)+'…' : s||''; }

async function startTraining() {
    var domain = document.getElementById('train-domain').value.trim();
    var mode   = document.getElementById('train-mode').value;
    if (!domain) { showStatus('train-status','Domain required.',false); return; }
    try {
        var d = await (await fetch('/api/nn/chain/train',{
            method:'POST', headers:{'Content-Type':'application/json'},
            body: JSON.stringify({Domain:domain, Mode:mode})
        })).json();
        showStatus('train-status','✓ '+d.result, true);
        setTimeout(loadJobs, 800);
    } catch(e) { showStatus('train-status','✗ '+e, false); }
}

async function loadJobs() {
    try {
        var jobs = await (await fetch('/api/nn/chain/jobs')).json();
        var el = document.getElementById('training-jobs-list');
        if (!jobs||!jobs.length) { el.innerHTML='<div class="empty-state">No training jobs yet.</div>'; return; }
        var html = jobs.slice().reverse().map(function(j) {
            var pct = j.total>0 ? Math.round(j.progress/j.total*100) : 0;
            var colMap = {done:'#6a9955',error:'#f44747',running:'#dcdcaa',queued:'#888'};
            var col = colMap[j.status]||'#888';
            return '<div style="background:#1a1a2a;border:1px solid #2a2a3a;border-radius:6px;padding:10px;margin-bottom:8px;">'
                 + '<div style="display:flex;justify-content:space-between;margin-bottom:4px;">'
                 + '<span style="color:#9cdcfe;font-size:0.8rem;">'+esc(j.expert_name||j.expert_id)+'</span>'
                 + '<span style="color:'+col+';font-size:0.72rem;">'+esc(j.status)+'</span>'
                 + '</div>'
                 + '<div style="font-size:0.7rem;color:#888;margin-bottom:6px;">mode: '+esc(j.mode)+' · '+esc(j.message||'')+'</div>'
                 + (j.total>0 ? '<div style="background:#2d2d2d;border-radius:3px;height:5px;overflow:hidden;">'
                   + '<div style="width:'+pct+'%;height:100%;background:'+col+';transition:width 0.4s;"></div></div>'
                   + '<div style="font-size:0.65rem;color:#555;margin-top:2px;">'+j.progress+'/'+j.total+'</div>' : '')
                 + '</div>';
        }).join('');
        el.innerHTML = html;
    } catch(e) { document.getElementById('training-jobs-list').innerHTML = '<div style="color:#f44747;">'+e+'</div>'; }
}

function filterAudit(){
    var q=(document.getElementById('audit-search')||{value:''}).value.toLowerCase();
    var lvl=(document.getElementById('audit-level-filter')||{value:''}).value;
    var c=document.getElementById('audit-list');
    var f=logs.filter(function(l){
        var matchQ = !q || l.content.toLowerCase().includes(q) || l.level.toLowerCase().includes(q);
        var matchL = !lvl || l.level === lvl;
        return matchQ && matchL;
    });
    var cntEl=document.getElementById('audit-count');
    if(cntEl) cntEl.textContent=f.length+' / '+logs.length+' entries';
    if(!f.length){c.innerHTML='<div class="empty-state">No matching entries.</div>';return;}
    var h='';
    f.slice().reverse().forEach(function(l){
        var ts=new Date(l.timestamp).toLocaleString();
        h+='<div class="audit-entry log-'+l.level+'" title="'+esc(ts)+'">'
         +'<span style="color:#666;font-size:0.7rem;margin-right:6px;">'+ts+'</span>'
         +'<b class="audit-lvl">'+esc(l.level)+'</b> '
         +esc(l.content)+'</div>';
    });
    c.innerHTML=h;
}

function printAuditLog(){
    var q=(document.getElementById('audit-search')||{value:''}).value.toLowerCase();
    var lvl=(document.getElementById('audit-level-filter')||{value:''}).value;
    var f=logs.filter(function(l){
        return (!q || l.content.toLowerCase().includes(q) || l.level.toLowerCase().includes(q))
            && (!lvl || l.level===lvl);
    });
    var lines=f.map(function(l){
        return new Date(l.timestamp).toLocaleString()+'\t'+l.level+'\t'+l.content;
    }).join('\n');
    var w=window.open('','_blank');
    w.document.write('<html><head><title>Audit Log</title>'
        +'<style>body{font-family:monospace;font-size:12px;padding:20px;white-space:pre-wrap;}</style></head>'
        +'<body>AUDIT LOG — '+new Date().toLocaleString()+' ('+f.length+' entries)\n\n'+lines+'</body></html>');
    w.document.close();
    w.print();
}

function exportAuditCSV(){
    var q=(document.getElementById('audit-search')||{value:''}).value.toLowerCase();
    var lvl=(document.getElementById('audit-level-filter')||{value:''}).value;
    var f=logs.filter(function(l){
        return (!q || l.content.toLowerCase().includes(q) || l.level.toLowerCase().includes(q))
            && (!lvl || l.level===lvl);
    });
    var csv='timestamp,level,content\n'+f.map(function(l){
        return '"'+new Date(l.timestamp).toISOString()+'","'+l.level+'","'+l.content.replace(/"/g,'""')+'"';
    }).join('\n');
    var a=document.createElement('a');
    a.href='data:text/csv;charset=utf-8,'+encodeURIComponent(csv);
    a.download='audit_log_'+Date.now()+'.csv';
    a.click();
}

/* ---- Neural Networks ---- */
async function loadNeuralNets(){
    try{
        var nets=await(await fetch('/api/nn/list')).json();
        var el=document.getElementById('nn-nets-list');
        if(!nets||!nets.length){el.innerHTML='<div class="empty-state" style="padding:10px;">No networks active yet. The default network (agent_memory) spawns automatically when the agent starts.</div>';return;}
        var h='<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:10px;">';
        nets.forEach(function(n){
            var pct=Math.round(n.util_pct||0);
            var color=pct>80?'#f44747':pct>50?'#dcdcaa':'#6a9955';
            h+='<div style="background:#1a1a2a;border:1px solid #3a3a5a;border-radius:6px;padding:10px;">'
             +'<div style="font-weight:bold;color:#9cdcfe;margin-bottom:4px;">🧠 '+esc(n.name)+'</div>'
             +'<div style="font-size:0.72rem;color:#888;margin-bottom:6px;">stored: '+n.stored+'/'+n.capacity+' &nbsp; recalls: '+n.recalls+'</div>'
             +'<div style="background:#2d2d2d;border-radius:3px;height:6px;margin-bottom:6px;overflow:hidden;">'
             +'<div style="width:'+pct+'%;height:100%;background:'+color+';transition:width 0.3s;"></div></div>'
             +'<div style="font-size:0.68rem;color:#666;">updated: '+esc(n.updated)+'</div>'
             +'</div>';
        });
        h+='</div>';
        el.innerHTML=h;
    }catch(e){document.getElementById('nn-nets-list').innerHTML='<div style="color:#f44747">Error loading networks: '+e+'</div>';}
}

async function nnSpawn(){
    var name=document.getElementById('nn-spawn-name').value.trim();
    var cap=parseInt(document.getElementById('nn-spawn-cap').value)||500;
    if(!name){showStatus('nn-spawn-status','Name required.',false);return;}
    try{
        var d=await(await fetch('/api/nn/spawn',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:name,capacity:cap})})).json();
        showStatus('nn-spawn-status','✓ '+d.result,true);
        loadNeuralNets();
    }catch(e){showStatus('nn-spawn-status','✗ '+e,false);}
}

async function nnTestRecall(){
    var net=(document.getElementById('nn-recall-net').value.trim()||'agent_memory');
    var q=document.getElementById('nn-recall-query').value.trim();
    var k=parseInt(document.getElementById('nn-recall-k').value)||3;
    if(!q){document.getElementById('nn-recall-result').textContent='Query required.';return;}
    try{
        var d=await(await fetch('/api/nn/recall',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({network:net,query:q,top_k:k})})).json();
        document.getElementById('nn-recall-result').textContent=d.result;
    }catch(e){document.getElementById('nn-recall-result').textContent='Error: '+e;}
}

async function nnManualStore(){
    var net=(document.getElementById('nn-store-net').value.trim()||'agent_memory');
    var key=document.getElementById('nn-store-key').value.trim();
    var val=document.getElementById('nn-store-val').value.trim();
    if(!key||!val){document.getElementById('nn-store-result').textContent='Key and value required.';return;}
    try{
        var d=await(await fetch('/api/nn/store',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({network:net,key:key,value:val})})).json();
        document.getElementById('nn-store-result').textContent=d.result;
        loadNeuralNets();
    }catch(e){document.getElementById('nn-store-result').textContent='Error: '+e;}
}


/* ---- Personality ---- */
async function loadPersonality(){
    try{ var d=await(await fetch('/api/personality')).json(); document.getElementById('personality-text').value=d.personality||''; }catch(e){}
}
async function savePersonality(){
    var text=document.getElementById('personality-text').value;
    try{
        await fetch('/api/personality',{method:'POST',body:JSON.stringify({personality:text}),headers:{'Content-Type':'application/json'}});
        var s=document.getElementById('personality-status');
        s.textContent='Saved. Takes effect on next task.';
        setTimeout(function(){s.textContent='';},3000);
    }catch(e){}
}

/* ---- Generic LLM config helpers ---- */
function updateBadge(badgeId, textId, url, model){
    var badge=document.getElementById(badgeId);
    var text=document.getElementById(textId);
    if(url&&model){ badge.className='llm-status-badge badge-ok'; text.textContent='Configured: '+model; }
    else           { badge.className='llm-status-badge badge-warn'; text.textContent='Not configured'; }
}
function updateHeaderBadge(id, cls, label, model){
    var el=document.getElementById(id);
    el.className='header-badge '+(model?cls:'hb-off');
    el.textContent=label+(model?' '+model:'—');
}
function showStatus(id, msg, ok){
    var el=document.getElementById(id);
    el.textContent=msg; el.style.color=ok?'#6a9955':'#f44747';
    setTimeout(function(){el.textContent='';el.style.color='';},5000);
}

/* ---- Main LLM ---- */
async function loadMainLLM(){
    try{
        var d=await(await fetch('/api/main_llm_config')).json();
        document.getElementById('main-llm-url').value=d.url||'';
        document.getElementById('main-llm-model').value=d.model||'';
        document.getElementById('main-llm-key').value=d.key||'';
        document.getElementById('main-llm-timeout').value=d.timeout||300;
        updateBadge('main-llm-badge','main-llm-badge-text',d.url,d.model);
        updateHeaderBadge('hdr-main','hb-main','Main: ',d.model);
        loadLLMHistory('main');
    }catch(e){}
}
async function saveMainLLM(){
    var url=document.getElementById('main-llm-url').value.trim();
    var model=document.getElementById('main-llm-model').value.trim();
    var key=document.getElementById('main-llm-key').value.trim();
    var timeout=parseInt(document.getElementById('main-llm-timeout').value)||300;
    try{
        await fetch('/api/main_llm_config',{method:'POST',body:JSON.stringify({url:url,model:model,key:key,timeout:timeout}),headers:{'Content-Type':'application/json'}});
        showStatus('main-llm-status','✓ Saved successfully.',true);
        updateBadge('main-llm-badge','main-llm-badge-text',url,model);
        updateHeaderBadge('hdr-main','hb-main','Main: ',model);
    }catch(e){ showStatus('main-llm-status','✗ Save failed: '+e,false); }
}
async function testMainLLM(){
    document.getElementById('main-llm-status').textContent='⏳ Testing...';
    document.getElementById('main-llm-status').style.color='#ce9178';
    document.getElementById('main-llm-test-result').style.display='none';
    try{
        var d=await(await fetch('/api/main_llm_test',{method:'POST'})).json();
        if(d.error){ showStatus('main-llm-status','✗ '+d.error,false); }
        else{
            showStatus('main-llm-status','✓ Connection successful!',true);
            var r=document.getElementById('main-llm-test-result');
            r.textContent=d.response; r.style.display='block';
        }
    }catch(e){ showStatus('main-llm-status','✗ Request failed: '+e,false); }
}

/* ---- Coding LLM ---- */
async function loadCodingLLM(){
    try{
        var d=await(await fetch('/api/coding_llm_config')).json();
        document.getElementById('coding-llm-url').value=d.url||'';
        document.getElementById('coding-llm-model').value=d.model||'';
        document.getElementById('coding-llm-key').value=d.key||'';
        document.getElementById('coding-llm-timeout').value=d.timeout||300;
        updateBadge('coding-llm-badge','coding-llm-badge-text',d.url,d.model);
        updateHeaderBadge('hdr-coding','hb-coding','Code: ',d.model);
    }catch(e){}
}
async function saveCodingLLM(){
    var url=document.getElementById('coding-llm-url').value.trim();
    var model=document.getElementById('coding-llm-model').value.trim();
    var key=document.getElementById('coding-llm-key').value.trim();
    var timeout=parseInt(document.getElementById('coding-llm-timeout').value)||300;
    try{
        await fetch('/api/coding_llm_config',{method:'POST',body:JSON.stringify({url:url,model:model,key:key,timeout:timeout}),headers:{'Content-Type':'application/json'}});
        showStatus('coding-llm-status','✓ Saved successfully.',true);
        updateBadge('coding-llm-badge','coding-llm-badge-text',url,model);
        updateHeaderBadge('hdr-coding','hb-coding','Code: ',model);
    }catch(e){ showStatus('coding-llm-status','✗ Save failed: '+e,false); }
}
async function testCodingLLM(){
    document.getElementById('coding-llm-status').textContent='⏳ Testing...';
    document.getElementById('coding-llm-status').style.color='#ce9178';
    document.getElementById('coding-llm-test-result').style.display='none';
    try{
        var d=await(await fetch('/api/coding_llm_test',{method:'POST'})).json();
        if(d.error){ showStatus('coding-llm-status','✗ '+d.error,false); }
        else{
            showStatus('coding-llm-status','✓ Connection successful!',true);
            var r=document.getElementById('coding-llm-test-result');
            r.textContent=d.response; r.style.display='block';
        }
    }catch(e){ showStatus('coding-llm-status','✗ Request failed: '+e,false); }
}

/* ---- Media LLM ---- */
async function loadMediaLLM(){
    try{
        var d=await(await fetch('/api/media_llm_config')).json();
        document.getElementById('media-llm-url').value=d.url||'';
        document.getElementById('media-llm-model').value=d.model||'';
        document.getElementById('media-llm-key').value=d.key||'';
        document.getElementById('media-llm-timeout').value=d.timeout||300;
        updateBadge('media-llm-badge','media-llm-badge-text',d.url,d.model);
        updateHeaderBadge('hdr-media','hb-media','Media: ',d.model);
    }catch(e){}
}
async function saveMediaLLM(){
    var url=document.getElementById('media-llm-url').value.trim();
    var model=document.getElementById('media-llm-model').value.trim();
    var key=document.getElementById('media-llm-key').value.trim();
    var timeout=parseInt(document.getElementById('media-llm-timeout').value)||300;
    try{
        await fetch('/api/media_llm_config',{method:'POST',body:JSON.stringify({url:url,model:model,key:key,timeout:timeout}),headers:{'Content-Type':'application/json'}});
        showStatus('media-llm-status','✓ Saved successfully.',true);
        updateBadge('media-llm-badge','media-llm-badge-text',url,model);
        updateHeaderBadge('hdr-media','hb-media','Media: ',model);
    }catch(e){ showStatus('media-llm-status','✗ Save failed: '+e,false); }
}
async function testMediaLLM(){
    document.getElementById('media-llm-status').textContent='⏳ Testing...';
    document.getElementById('media-llm-status').style.color='#ce9178';
    document.getElementById('media-llm-test-result').style.display='none';
    try{
        var d=await(await fetch('/api/media_llm_test',{method:'POST'})).json();
        if(d.error){ showStatus('media-llm-status','✗ '+d.error,false); }
        else{
            showStatus('media-llm-status','✓ Connection successful!',true);
            var r=document.getElementById('media-llm-test-result');
            r.textContent=d.response; r.style.display='block';
        }
    }catch(e){ showStatus('media-llm-status','✗ Request failed: '+e,false); }
}

/* ---- Content LLM ---- */
async function loadContentLLM(){
    try{
        var d=await(await fetch('/api/content_llm_config')).json();
        document.getElementById('content-llm-url').value=d.url||'';
        document.getElementById('content-llm-model').value=d.model||'';
        document.getElementById('content-llm-key').value=d.key||'';
        document.getElementById('content-llm-timeout').value=d.timeout||300;
        updateBadge('content-llm-badge','content-llm-badge-text',d.url,d.model);
        updateHeaderBadge('hdr-content','hb-content','Content: ',d.model);
    }catch(e){}
}
async function saveContentLLM(){
    var url=document.getElementById('content-llm-url').value.trim();
    var model=document.getElementById('content-llm-model').value.trim();
    var key=document.getElementById('content-llm-key').value.trim();
    var timeout=parseInt(document.getElementById('content-llm-timeout').value)||300;
    try{
        await fetch('/api/content_llm_config',{method:'POST',body:JSON.stringify({url:url,model:model,key:key,timeout:timeout}),headers:{'Content-Type':'application/json'}});
        showStatus('content-llm-status','✓ Saved successfully.',true);
        updateBadge('content-llm-badge','content-llm-badge-text',url,model);
        updateHeaderBadge('hdr-content','hb-content','Content: ',model);
    }catch(e){ showStatus('content-llm-status','✗ Save failed: '+e,false); }
}
async function testContentLLM(){
    document.getElementById('content-llm-status').textContent='⏳ Testing...';
    document.getElementById('content-llm-status').style.color='#ce9178';
    document.getElementById('content-llm-test-result').style.display='none';
    try{
        var d=await(await fetch('/api/content_llm_test',{method:'POST'})).json();
        if(d.error){ showStatus('content-llm-status','✗ '+d.error,false); }
        else{
            showStatus('content-llm-status','✓ Connection successful!',true);
            var r=document.getElementById('content-llm-test-result');
            r.textContent=d.response; r.style.display='block';
        }
    }catch(e){ showStatus('content-llm-status','✗ Request failed: '+e,false); }
}

/* ---- New chat / send task ---- */
async function newChat(){
    if(!logs||!logs.length){alert('Nothing in the current chat to save.');return;}
    await fetch('/api/new_chat',{method:'POST'});
    logs=[];
    document.getElementById('chat-output').innerHTML='<div class="empty-state">New chat started. Previous session saved.</div>';
    filterAudit();
    sessions=await(await fetch('/api/sessions')).json();
    renderSessions(); switchTab('sessions');
}
async function sendTask(){
    var inp=document.getElementById('user-input');
    var txt=inp.value.trim();
    if(!txt) return;
    inp.value=''; inp.style.height='42px';
    showTyping(true);
    await fetch('/api/task',{method:'POST',body:JSON.stringify({task:txt}),headers:{'Content-Type':'application/json'}});
    setTimeout(fetchChatMessages,200);
}
function handleEnter(e){
    if(e.key==='Enter'&&!e.shiftKey){ e.preventDefault(); sendTask(); }
}
// Auto-resize textarea
document.addEventListener('DOMContentLoaded',function(){
    var inp=document.getElementById('user-input');
    if(inp){ inp.addEventListener('input',function(){
        this.style.height='42px';
        this.style.height=Math.min(this.scrollHeight,120)+'px';
    }); inp.addEventListener('keypress',handleEnter); }
});
function esc(s){ return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

/* ---- Host / Root Access ---- */
async function loadHostConfig(){
    try{
        var d=await(await fetch('/api/host_config')).json();
        document.getElementById('host-user').value=d.user||'';
        document.getElementById('host-pass').value=d.pass||'';
        updateHostBadge(d.user, !!d.pass_set);
    }catch(e){}
}
function updateHostBadge(user, hasPass){
    var badge=document.getElementById('host-badge');
    var text=document.getElementById('host-badge-text');
    var hdr=document.getElementById('hdr-root');
    if(hasPass){
        badge.className='llm-status-badge badge-ok';
        text.textContent='Active: running as '+(user||'root');
        hdr.className='header-badge hb-root';
        hdr.textContent='Root: '+(user||'root');
    } else {
        badge.className='llm-status-badge badge-warn';
        text.textContent='No credentials set';
        hdr.className='header-badge hb-off';
        hdr.textContent='Root: off';
    }
}
async function saveHostConfig(){
    var user=document.getElementById('host-user').value.trim();
    var pass=document.getElementById('host-pass').value;
    try{
        var r=await fetch('/api/host_config',{method:'POST',body:JSON.stringify({user:user,pass:pass}),headers:{'Content-Type':'application/json'}});
        var d=await r.json();
        showStatus('host-status','✓ Credentials saved.',true);
        updateHostBadge(user, pass.length>0);
    }catch(e){ showStatus('host-status','✗ Save failed: '+e,false); }
}
async function clearHostConfig(){
    if(!confirm('Clear host credentials? The agent will no longer run as root.')) return;
    try{
        await fetch('/api/host_config',{method:'POST',body:JSON.stringify({user:'',pass:''}),headers:{'Content-Type':'application/json'}});
        document.getElementById('host-user').value='';
        document.getElementById('host-pass').value='';
        showStatus('host-status','✓ Credentials cleared.',true);
        updateHostBadge('',false);
    }catch(e){ showStatus('host-status','✗ Failed: '+e,false); }
}
async function verifyHostAccess(){
    showStatus('host-status','⏳ Verifying...',true);
    document.getElementById('host-status').style.color='#ce9178';
    document.getElementById('host-verify-result').style.display='none';
    try{
        var r=await fetch('/api/host_verify',{method:'POST'});
        var d=await r.json();
        if(d.error){ showStatus('host-status','✗ '+d.error,false); }
        else{
            showStatus('host-status','✓ Access verified!',true);
            var el=document.getElementById('host-verify-result');
            el.textContent=d.output; el.style.display='block';
        }
    }catch(e){ showStatus('host-status','✗ Request failed: '+e,false); }
}

/* ---- UI Title ---- */
async function loadUITitle(){
    try{
        var d=await(await fetch('/api/ui_title')).json();
        var t=d.title||'Lily';
        document.getElementById('ui-title-input').value=t;
        document.title=t;
        document.getElementById('header-title').textContent=t;
    }catch(e){}
}
async function saveUITitle(){
    var t=document.getElementById('ui-title-input').value.trim()||'Lily';
    try{
        await fetch('/api/ui_title',{method:'POST',body:JSON.stringify({title:t}),headers:{'Content-Type':'application/json'}});
        document.title=t;
        document.getElementById('header-title').textContent=t;
        var s=document.getElementById('ui-title-status');
        s.textContent='✓ Saved'; s.style.color='#6a9955';
        setTimeout(function(){s.textContent='';},3000);
    }catch(e){
        var s=document.getElementById('ui-title-status');
        s.textContent='✗ Failed'; s.style.color='#f44747';
    }
}

/* ---- Tasks ---- */
var allTasks=[], taskFilter='all', editingTaskId=null;

function setTaskFilter(f, el){
    taskFilter=f;
    document.querySelectorAll('.filter-btn').forEach(function(b){ b.classList.remove('active'); });
    el.classList.add('active');
    renderTasks();
}

function renderTasks(){
    var c=document.getElementById('task-list');
    var f=taskFilter==='all' ? allTasks : allTasks.filter(function(t){return t.status===taskFilter;});
    if(!f.length){ c.innerHTML='<div class="empty-state">No tasks match.</div>'; return; }
    // update stats
    ['pending','running','complete','failed'].forEach(function(s){
        var el=document.getElementById('stat-'+s);
        if(el) el.textContent=allTasks.filter(function(t){return t.status===s;}).length;
    });
    var h='';
    f.slice().reverse().forEach(function(t){
        var pLabel=['','🔴 High','🟡 Normal','🟢 Low'][t.priority]||'';
        var pClass=['','tb-p1','tb-p2','tb-p3'][t.priority]||'';
        if(editingTaskId===t.id){
            h+='<div class="task-edit-row" id="edit-row-'+t.id+'">'
              +'<input type="text" id="et-title-'+t.id+'" value="'+esc(t.title)+'" placeholder="Title">'
              +'<textarea id="et-desc-'+t.id+'" placeholder="Description">'+esc(t.description||'')+'</textarea>'
              +'<div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;">'
              +'<select id="et-status-'+t.id+'">'
              +['pending','running','complete','failed'].map(function(s){
                  return '<option value="'+s+'"'+(t.status===s?' selected':'')+'>'+s+'</option>';
              }).join('')+'</select>'
              +'<select id="et-prio-'+t.id+'">'
              +[['1','🔴 High'],['2','🟡 Normal'],['3','🟢 Low']].map(function(p){
                  return '<option value="'+p[0]+'"'+(t.priority==p[0]?' selected':'')+'>'+p[1]+'</option>';
              }).join('')+'</select>'
              +'</div>'
              +'<input type="text" id="et-result-'+t.id+'" value="'+esc(t.result||'')+'" placeholder="Result / notes">'
              +'<div class="task-edit-btns">'
              +'<button class="btn btn-green" style="padding:5px 12px;font-size:0.78rem;" onclick="saveTaskEdit('+t.id+')">💾 Save</button>'
              +'<button class="btn btn-amber" style="padding:5px 12px;font-size:0.78rem;" onclick="cancelTaskEdit()">Cancel</button>'
              +'</div></div>';
        } else {
            h+='<div class="task-item task-'+t.status+'">'
              +'<div class="task-title"><span class="task-badge tb-'+t.status+'">'+t.status.toUpperCase()+'</span> '+esc(t.title)+'</div>';
            if(t.description) h+='<div class="task-desc">'+esc(t.description)+'</div>';
            if(t.result) h+='<div class="task-result">→ '+esc(t.result)+'</div>';
            h+='<div class="task-meta"><span class="'+pClass+'">'+pLabel+'</span>'
              +'<span>#'+t.id+' · '+new Date(t.created_at).toLocaleString()+'</span></div>'
              +'<div class="task-actions">'
              +'<button class="task-act-btn tact-edit" onclick="startTaskEdit('+t.id+')">✏ Edit</button>'
              +'<button class="task-act-btn tact-status" onclick="cycleStatus('+t.id+',\''+t.status+'\')" title="Change status">⟳ Status</button>'
              +'<button class="task-act-btn tact-delete" onclick="deleteTask('+t.id+')">✕ Delete</button>'
              +'</div></div>';
        }
    });
    c.innerHTML=h;
}

function startTaskEdit(id){
    editingTaskId=id;
    renderTasks();
    var el=document.getElementById('edit-row-'+id);
    if(el) el.scrollIntoView({block:'nearest'});
}
function cancelTaskEdit(){ editingTaskId=null; renderTasks(); }

async function saveTaskEdit(id){
    var title  =document.getElementById('et-title-'+id).value.trim();
    var desc   =document.getElementById('et-desc-'+id).value.trim();
    var status =document.getElementById('et-status-'+id).value;
    var priority=parseInt(document.getElementById('et-prio-'+id).value);
    var result =document.getElementById('et-result-'+id).value.trim();
    if(!title) return;
    try{
        await fetch('/api/tasks/'+id,{method:'PUT',
            body:JSON.stringify({title:title,description:desc,status:status,priority:priority,result:result}),
            headers:{'Content-Type':'application/json'}});
        editingTaskId=null;
        fetchTasks();
    }catch(e){ alert('Save failed: '+e); }
}

async function deleteTask(id){
    if(!confirm('Delete this task?')) return;
    try{
        await fetch('/api/tasks/'+id,{method:'DELETE'});
        fetchTasks();
    }catch(e){ alert('Delete failed: '+e); }
}

var statusCycle=['pending','running','complete','failed'];
async function cycleStatus(id, current){
    var next=statusCycle[(statusCycle.indexOf(current)+1)%statusCycle.length];
    try{
        await fetch('/api/tasks/'+id,{method:'PUT',
            body:JSON.stringify({status:next}),
            headers:{'Content-Type':'application/json'}});
        fetchTasks();
    }catch(e){}
}

async function fetchTasks(){
    try{
        var d=await(await fetch('/api/tasks')).json();
        if(JSON.stringify(d)!==JSON.stringify(allTasks)){ allTasks=d||[]; renderTasks(); updateTaskBadge(); }
    }catch(e){}
}

function updateTaskBadge(){
    var pending=allTasks.filter(function(t){return t.status==='pending';}).length;
    var running=allTasks.filter(function(t){return t.status==='running';}).length;
    var hdr=document.getElementById('hdr-tasks');
    if(!hdr) return;
    if(running>0){ hdr.className='header-badge hb-task-run'; hdr.textContent='Tasks: '+running+' running'; }
    else if(pending>0){ hdr.className='header-badge hb-task-pend'; hdr.textContent='Tasks: '+pending+' pending'; }
    else{ hdr.className='header-badge hb-off'; hdr.textContent='Tasks: idle'; }
}

async function addTask(){
    var title=document.getElementById('task-title').value.trim();
    var desc=document.getElementById('task-desc').value.trim();
    var priority=parseInt(document.getElementById('task-priority').value);
    if(!title){ document.getElementById('task-add-status').textContent='Title is required.'; return; }
    try{
        var r=await fetch('/api/tasks',{method:'POST',
            body:JSON.stringify({title:title,description:desc,priority:priority}),
            headers:{'Content-Type':'application/json'}});
        var d=await r.json();
        document.getElementById('task-title').value='';
        document.getElementById('task-desc').value='';
        var s=document.getElementById('task-add-status');
        s.textContent='✓ Task #'+d.id+' created.'; s.style.color='#6a9955';
        setTimeout(function(){s.textContent='';},3000);
        fetchTasks();
    }catch(e){
        var s=document.getElementById('task-add-status');
        s.textContent='✗ Failed: '+e; s.style.color='#f44747';
    }
}

/* ---- Modules ---- */
function switchModTab(name, el){
    ['browse','installed','gomod'].forEach(function(t){
        document.getElementById('mod-pane-'+t).style.display='none';
    });
    document.getElementById('mod-pane-'+name).style.display='flex';
    document.getElementById('mod-pane-'+name).style.flexDirection='column';
    document.querySelectorAll('.mod-tab').forEach(function(b){ b.classList.remove('active'); });
    el.classList.add('active');
    if(name==='installed') loadInstalledModules();
    if(name==='gomod') loadGoMod();
}

async function searchModules(){
    var q=document.getElementById('mod-search-input').value.trim();
    if(!q) return;
    var s=document.getElementById('mod-search-status');
    s.textContent='⏳ Searching...'; s.style.color='#ce9178';
    document.getElementById('mod-results').innerHTML='';
    try{
        var r=await fetch('/api/modules/search?q='+encodeURIComponent(q));
        var d=await r.json();
        s.textContent=''; 
        renderModResults(d.results||[]);
    }catch(e){ s.textContent='✗ Search failed: '+e; s.style.color='#f44747'; }
}

function renderModResults(results){
    var c=document.getElementById('mod-results');
    if(!results.length){ c.innerHTML='<div class="empty-state">No results found.</div>'; return; }
    var h='';
    results.forEach(function(m){
        h+='<div class="mod-result">'
          +'<div class="mod-name">'+esc(m.path)+'</div>'
          +(m.synopsis?'<div class="mod-synopsis">'+esc(m.synopsis)+'</div>':'')
          +'<div class="mod-meta">'
          +(m.version?'<span class="mod-ver">'+esc(m.version)+'</span>':'')
          +(m.license?'<span class="mod-license">'+esc(m.license)+'</span>':'')
          +'<button class="mod-btn mod-btn-add" onclick="installModule(\''+esc(m.path)+'\',\'latest\')">+ Install</button>'
          +'</div></div>';
    });
    c.innerHTML=h;
}

async function installModule(path, version){
    var s=document.getElementById('mod-search-status');
    s.textContent='⏳ Installing '+path+'...'; s.style.color='#ce9178';
    try{
        var r=await fetch('/api/modules/add',{method:'POST',
            body:JSON.stringify({module_path:path,version:version||'latest'}),
            headers:{'Content-Type':'application/json'}});
        var d=await r.json();
        if(d.error){ s.textContent='✗ '+d.error; s.style.color='#f44747'; }
        else{ s.textContent='✓ Installed '+path; s.style.color='#6a9955'; }
        setTimeout(function(){s.textContent='';},4000);
    }catch(e){ s.textContent='✗ Failed: '+e; s.style.color='#f44747'; }
}

async function loadInstalledModules(){
    var c=document.getElementById('mod-installed-list');
    c.innerHTML='<div class="empty-state">Loading...</div>';
    try{
        var r=await fetch('/api/modules/installed');
        var d=await r.json();
        renderInstalledMods(d.modules||[]);
    }catch(e){ c.innerHTML='<div class="empty-state">Error loading modules.</div>'; }
}

function renderInstalledMods(mods){
    var c=document.getElementById('mod-installed-list');
    if(!mods.length){ c.innerHTML='<div class="empty-state">No dependencies found.</div>'; return; }
    var h='';
    mods.forEach(function(m){
        if(m.path==='') return;
        h+='<div class="installed-mod">'
          +'<span class="installed-mod-path">'+esc(m.path)+'</span>'
          +(m.version?'<span class="installed-mod-ver">'+esc(m.version)+'</span>':'')
          +'<button class="mod-btn mod-btn-rm" onclick="removeModuleUI(\''+esc(m.path)+'\')">✕</button>'
          +'</div>';
    });
    c.innerHTML=h;
}

async function removeModuleUI(path){
    if(!confirm('Remove '+path+'?')) return;
    var s=document.getElementById('mod-install-status');
    s.textContent='⏳ Removing...'; s.style.color='#ce9178';
    try{
        var r=await fetch('/api/modules/remove',{method:'POST',
            body:JSON.stringify({module_path:path}),
            headers:{'Content-Type':'application/json'}});
        var d=await r.json();
        if(d.error){ s.textContent='✗ '+d.error; s.style.color='#f44747'; }
        else{ s.textContent='✓ Removed '+path; s.style.color='#6a9955'; loadInstalledModules(); }
        setTimeout(function(){s.textContent='';},4000);
    }catch(e){ s.textContent='✗ Failed: '+e; s.style.color='#f44747'; }
}

async function addModuleManual(){
    var path=document.getElementById('mod-manual-input').value.trim();
    if(!path) return;
    installModule(path,'latest');
    document.getElementById('mod-manual-input').value='';
}

async function loadGoMod(){
    try{
        var r=await fetch('/api/modules/gomod');
        var d=await r.json();
        document.getElementById('mod-gomod-content').value=d.content||'(go.mod not found)';
    }catch(e){}
}

/* ---- LLM History ---- */
var llmHistoryCache = {main:[], coding:[], media:[], content:[]};

async function loadLLMHistory(slot){
    try{
        var d = await(await fetch('/api/llm_history?slot='+slot)).json();
        llmHistoryCache[slot] = d || [];
        var sel = document.getElementById('hist-'+slot);
        if(!sel) return;
        // Keep placeholder option, rebuild rest
        while(sel.options.length > 1) sel.remove(1);
        (d||[]).forEach(function(p, i){
            var opt = document.createElement('option');
            opt.value = i;
            opt.textContent = p.label || (p.model + ' @ ' + p.url);
            sel.appendChild(opt);
        });
    }catch(e){}
}

function applyLLMHistory(slot, idx){
    if(idx === '' || idx === null) return;
    var profile = llmHistoryCache[slot][parseInt(idx)];
    if(!profile) return;
    document.getElementById(slot+'-llm-url').value   = profile.url   || '';
    document.getElementById(slot+'-llm-model').value = profile.model || '';
    // Only fill key if not masked (not all asterisks)
    var k = profile.key || '';
    if(k && !/^\*+$/.test(k)) document.getElementById(slot+'-llm-key').value = k;
    var tout = document.getElementById(slot+'-llm-timeout');
    if(tout) tout.value = profile.timeout || 300;
    // Reset dropdown to placeholder after apply
    document.getElementById('hist-'+slot).value = '';
    // Show visual confirmation
    var badge = document.getElementById(slot+'-llm-badge-text');
    if(badge){ var old=badge.textContent; badge.textContent='✓ Profile loaded'; setTimeout(function(){badge.textContent=old;},1500); }
}

async function loadAllLLMHistories(){
    await Promise.all(['main','coding','media','content'].map(loadLLMHistory));
}

/* ---- Delegation ---- */
var delegationState = {coding: true, media: true, content: true};

async function loadDelegation(){
    try{
        var d = await(await fetch('/api/delegation')).json();
        delegationState = d;
        applyDelegationUI();
    }catch(e){}
}

function applyDelegationUI(){
    ['coding','media','content'].forEach(function(k){
        var chk = document.getElementById('toggle-'+k);
        var lbl = document.getElementById(k+'-toggle-label');
        var bar = document.getElementById(k+'-delegation-bar');
        var on  = delegationState[k];
        if(chk) chk.checked = on;
        if(lbl){ lbl.textContent = on ? 'ON' : 'OFF'; lbl.className = 'toggle-label '+(on?'on':'off'); }
        if(bar){ bar.className = 'delegation-bar'+(on?'':' disabled'); }
    });
}

async function toggleDelegation(key, enabled){
    delegationState[key] = enabled;
    applyDelegationUI();
    var payload = {};
    payload[key] = enabled;
    try{
        await fetch('/api/delegation',{method:'POST',
            body: JSON.stringify(payload),
            headers:{'Content-Type':'application/json'}});
    }catch(e){ console.error('delegation save failed', e); }
}

/* ---- Web Search ---- */
var selectedProvider='duckduckgo';

function selectProvider(p){
    selectedProvider=p;
    document.querySelectorAll('.provider-card').forEach(function(c){ c.classList.remove('selected'); });
    document.getElementById('prov-'+p).classList.add('selected');
    var kf=document.getElementById('search-key-fields');
    var cxg=document.getElementById('search-cx-group');
    var hint=document.getElementById('search-key-hint');
    if(p==='duckduckgo'){ kf.style.display='none'; cxg.style.display='none'; }
    else if(p==='google'){
        kf.style.display='block'; cxg.style.display='block';
        if(hint) hint.textContent='Google Custom Search API key from console.developers.google.com';
    } else {
        kf.style.display='block'; cxg.style.display='none';
        if(hint) hint.textContent='Brave Search API key from api.search.brave.com';
    }
}

async function loadSearchConfig(){
    try{
        var d=await(await fetch('/api/websearch_config')).json();
        var p=d.provider||'duckduckgo';
        selectProvider(p);
        document.getElementById('search-api-key').value=d.key||'';
        document.getElementById('search-cx').value=d.cx||'';
    }catch(e){}
}

async function saveSearchConfig(){
    var key=document.getElementById('search-api-key').value.trim();
    var cx=document.getElementById('search-cx').value.trim();
    try{
        await fetch('/api/websearch_config',{method:'POST',
            body:JSON.stringify({provider:selectedProvider,key:key,cx:cx}),
            headers:{'Content-Type':'application/json'}});
        showStatus('search-status','✓ Saved.',true);
    }catch(e){ showStatus('search-status','✗ '+e,false); }
}

async function testSearch(){
    var q=document.getElementById('search-test-query').value.trim()||'latest AI news';
    showStatus('search-status','⏳ Searching...',true);
    document.getElementById('search-status').style.color='#ce9178';
    var box=document.getElementById('search-test-box');
    box.style.display='none';
    try{
        var r=await fetch('/api/websearch_test',{method:'POST',
            body:JSON.stringify({query:q}),headers:{'Content-Type':'application/json'}});
        var d=await r.json();
        showStatus('search-status','✓ Results received.',true);
        box.textContent=d.result||'No results.'; box.style.display='block';
    }catch(e){ showStatus('search-status','✗ '+e,false); }
}

/* ---- Social ---- */
async function loadSocialConfig(){
    try{
        var d=await(await fetch('/api/social_config')).json();
        document.getElementById('soc-discord-token').value=d.discord_token||'';
        document.getElementById('soc-discord-channel').value=d.discord_channel||'';
        document.getElementById('soc-twitter-key').value=d.twitter_key||'';
        document.getElementById('soc-twitter-secret').value=d.twitter_secret||'';
        document.getElementById('soc-twitter-token').value=d.twitter_token||'';
        document.getElementById('soc-twitter-token-sec').value=d.twitter_token_sec||'';
        document.getElementById('soc-telegram-token').value=d.telegram_token||'';
        document.getElementById('soc-telegram-chat').value=d.telegram_chat_id||'';
        document.getElementById('soc-slack-token').value=d.slack_token||'';
        document.getElementById('soc-slack-channel').value=d.slack_channel||'';
        document.getElementById('soc-wa-sid').value=d.whatsapp_sid||'';
        document.getElementById('soc-wa-token').value=d.whatsapp_token||'';
        document.getElementById('soc-wa-from').value=d.whatsapp_from||'';
        document.getElementById('soc-wa-to').value=d.whatsapp_to||'';
        // update dots
        ['discord','twitter','telegram','slack','whatsapp'].forEach(function(p){
            var dot=document.getElementById('dot-'+p);
            if(!dot) return;
            var hasKey=false;
            if(p==='discord')  hasKey=(d.discord_token||'').length>4;
            if(p==='twitter')  hasKey=(d.twitter_key||'').length>4;
            if(p==='telegram') hasKey=(d.telegram_token||'').length>4;
            if(p==='slack')    hasKey=(d.slack_token||'').length>4;
            if(p==='whatsapp') hasKey=(d.whatsapp_sid||'').length>4;
            dot.classList.toggle('on', hasKey);
        });
    }catch(e){}
}

async function saveSocialConfig(){
    var payload={
        discord_token: document.getElementById('soc-discord-token').value,
        discord_channel: document.getElementById('soc-discord-channel').value,
        twitter_key: document.getElementById('soc-twitter-key').value,
        twitter_secret: document.getElementById('soc-twitter-secret').value,
        twitter_token: document.getElementById('soc-twitter-token').value,
        twitter_token_sec: document.getElementById('soc-twitter-token-sec').value,
        telegram_token: document.getElementById('soc-telegram-token').value,
        telegram_chat_id: document.getElementById('soc-telegram-chat').value,
        slack_token: document.getElementById('soc-slack-token').value,
        slack_channel: document.getElementById('soc-slack-channel').value,
        whatsapp_sid: document.getElementById('soc-wa-sid').value,
        whatsapp_token: document.getElementById('soc-wa-token').value,
        whatsapp_from: document.getElementById('soc-wa-from').value,
        whatsapp_to: document.getElementById('soc-wa-to').value,
    };
    try{
        await fetch('/api/social_config',{method:'POST',body:JSON.stringify(payload),headers:{'Content-Type':'application/json'}});
        showStatus('social-save-status','✓ All credentials saved.',true);
        loadSocialConfig();
    }catch(e){ showStatus('social-save-status','✗ '+e,false); }
}

async function testSocial(platform){
    var s=document.getElementById('soc-status-'+platform);
    if(s){ s.textContent='⏳ Sending...'; s.style.color='#ce9178'; }
    try{
        var r=await fetch('/api/social_test',{method:'POST',
            body:JSON.stringify({platform:platform,message:'✅ Test from agent — '+new Date().toLocaleTimeString()}),
            headers:{'Content-Type':'application/json'}});
        var d=await r.json();
        if(s){
            s.textContent=d.ok?'✓ '+d.result:'✗ '+d.result;
            s.style.color=d.ok?'#6a9955':'#f44747';
            setTimeout(function(){s.textContent='';s.style.color='';},5000);
        }
    }catch(e){ if(s){ s.textContent='✗ '+e; s.style.color='#f44747'; } }
}

/* ---- Init ---- */
loadPersonality();
loadUITitle();
loadHostConfig();
loadDelegation();
loadNeuralNets();
loadChain();
loadAllLLMHistories();
loadSearchConfig();
loadSocialConfig();
loadMainLLM();
loadCodingLLM();
loadMediaLLM();
loadContentLLM();
setInterval(fetchData,1000);
fetchData();

</script>
</body>
</html>`

func startWebServer() {
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		title := webUITitle
		mu.Unlock()
		page := strings.ReplaceAll(HTML_TEMPLATE, "__TITLE__", title)
		fmt.Fprint(w, page)
	})

	http.HandleFunc("/api/chat_messages", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		mu.Lock()
		msgs := make([]ChatMessage, len(chatMessages))
		copy(msgs, chatMessages)
		mu.Unlock()
		if msgs == nil {
			msgs = []ChatMessage{}
		}
		json.NewEncoder(w).Encode(msgs)
	})

	http.HandleFunc("/api/task", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			http.Error(w, "Invalid method", 405)
			return
		}
		var req struct {
			Task string `json:"task"`
		}
		json.NewDecoder(r.Body).Decode(&req)
		if req.Task != "" {
			taskQueue <- req.Task
			mu.Lock()
			chatMessages = append(chatMessages, ChatMessage{Role: "user", Content: req.Task, Timestamp: time.Now()})
			chatTask := Task{
				ID:          len(tasksDb) + 1,
				Title:       truncate(req.Task, 80),
				Description: req.Task,
				Status:      "running",
				Priority:    2,
				CreatedAt:   time.Now(),
				UpdatedAt:   time.Now(),
			}
			tasksDb = append(tasksDb, chatTask)
			saveTasksLocked()
			mu.Unlock()
			auditLog("TASK", fmt.Sprintf("Chat task #%d auto-created: %s", chatTask.ID, chatTask.Title))
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"status": "queued"})
	})

	http.HandleFunc("/api/logs", func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		defer mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(auditLogs)
	})

	http.HandleFunc("/api/memory", func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		defer mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(memoryDb)
	})

	http.HandleFunc("/api/sessions", func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		defer mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(chatSessions)
	})

	http.HandleFunc("/api/new_chat", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			http.Error(w, "Invalid method", 405)
			return
		}
		session := newChat()
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(session)
		auditLog("SYSTEM", fmt.Sprintf("New chat started. Session #%d archived.", session.ID))
	})

	http.HandleFunc("/api/personality", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.Method == "GET" {
			mu.Lock()
			p := currentPersonality
			mu.Unlock()
			json.NewEncoder(w).Encode(map[string]string{"personality": p})
			return
		}
		if r.Method == "POST" {
			var req struct {
				Personality string `json:"personality"`
			}
			json.NewDecoder(r.Body).Decode(&req)
			mu.Lock()
			currentPersonality = req.Personality
			saveConfigLocked()
			mu.Unlock()
			go nnCtrl.store(NN_IDENTITY, "agent_personality", req.Personality)
			auditLog("CONFIG", "Personality updated and stored in NN identity network.")
			json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
			return
		}
		http.Error(w, "Invalid method", 405)
	})

	http.HandleFunc("/api/main_llm_config", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.Method == "GET" {
			mu.Lock()
			u, m, k, t := mainLLMURL, mainLLMModel, mainLLMKey, llmTimeoutSec
			mu.Unlock()
			json.NewEncoder(w).Encode(map[string]interface{}{"url": u, "model": m, "key": maskKey(k), "timeout": t})
			return
		}
		if r.Method == "POST" {
			var req struct {
				URL     string `json:"url"`
				Model   string `json:"model"`
				Key     string `json:"key"`
				Timeout int    `json:"timeout"`
			}
			json.NewDecoder(r.Body).Decode(&req)
			mu.Lock()
			mainLLMURL = strings.TrimRight(req.URL, "/")
			mainLLMModel = req.Model
			if req.Key != "" && !isAllAsterisks(req.Key) {
				mainLLMKey = req.Key
			}
			if req.Timeout > 0 {
				llmTimeoutSec = req.Timeout
			}
			pushLLMHistory("main", mainLLMURL, mainLLMModel, mainLLMKey, llmTimeoutSec)
			saveConfigLocked()
			mu.Unlock()
			auditLog("CONFIG", fmt.Sprintf("Main LLM updated: model=%s url=%s timeout=%ds", req.Model, req.URL, req.Timeout))
			go nnCtrl.store(NN_SEMANTIC, "main_llm_config", fmt.Sprintf("model=%s url=%s", req.Model, req.URL))
			json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
			return
		}
		http.Error(w, "Invalid method", 405)
	})

	http.HandleFunc("/api/main_llm_test", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			http.Error(w, "Invalid method", 405)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		result, err := chatWithMainLLM([]Message{
			{Role: "system", Content: "You are a helpful assistant."},
			{Role: "user", Content: "Reply with exactly one sentence confirming you are online and state your model name."},
		})
		if err != nil {
			json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
			return
		}
		auditLog("CONFIG", "Main LLM test connection successful.")
		json.NewEncoder(w).Encode(map[string]string{"response": result})
	})

	http.HandleFunc("/api/coding_llm_config", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.Method == "GET" {
			mu.Lock()
			u, m, k, t := codingLLMURL, codingLLMModel, codingLLMKey, llmTimeoutSec
			mu.Unlock()
			json.NewEncoder(w).Encode(map[string]interface{}{"url": u, "model": m, "key": maskKey(k), "timeout": t})
			return
		}
		if r.Method == "POST" {
			var req struct {
				URL     string `json:"url"`
				Model   string `json:"model"`
				Key     string `json:"key"`
				Timeout int    `json:"timeout"`
			}
			json.NewDecoder(r.Body).Decode(&req)
			mu.Lock()
			codingLLMURL = strings.TrimRight(req.URL, "/")
			codingLLMModel = req.Model
			if req.Key != "" && !isAllAsterisks(req.Key) {
				codingLLMKey = req.Key
			}
			if req.Timeout > 0 {
				llmTimeoutSec = req.Timeout
			}
			pushLLMHistory("coding", codingLLMURL, codingLLMModel, codingLLMKey, llmTimeoutSec)
			saveConfigLocked()
			mu.Unlock()
			auditLog("CONFIG", fmt.Sprintf("Coding LLM updated: model=%s url=%s timeout=%ds", req.Model, req.URL, req.Timeout))
			go nnCtrl.store(NN_SEMANTIC, "coding_llm_config", fmt.Sprintf("model=%s url=%s", req.Model, req.URL))
			json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
			return
		}
		http.Error(w, "Invalid method", 405)
	})

	http.HandleFunc("/api/coding_llm_test", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			http.Error(w, "Invalid method", 405)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		result, err := chatWithCodingLLM("Reply with exactly one sentence confirming you are online and state your model name.")
		if err != nil {
			json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
			return
		}
		auditLog("CODING_LLM", "Test connection successful.")
		json.NewEncoder(w).Encode(map[string]string{"response": result})
	})

	http.HandleFunc("/api/media_llm_config", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.Method == "GET" {
			mu.Lock()
			u, m, k, t := mediaLLMURL, mediaLLMModel, mediaLLMKey, llmTimeoutSec
			mu.Unlock()
			json.NewEncoder(w).Encode(map[string]interface{}{"url": u, "model": m, "key": maskKey(k), "timeout": t})
			return
		}
		if r.Method == "POST" {
			var req struct {
				URL     string `json:"url"`
				Model   string `json:"model"`
				Key     string `json:"key"`
				Timeout int    `json:"timeout"`
			}
			json.NewDecoder(r.Body).Decode(&req)
			mu.Lock()
			mediaLLMURL = strings.TrimRight(req.URL, "/")
			mediaLLMModel = req.Model
			if req.Key != "" && !isAllAsterisks(req.Key) {
				mediaLLMKey = req.Key
			}
			if req.Timeout > 0 {
				llmTimeoutSec = req.Timeout
			}
			pushLLMHistory("media", mediaLLMURL, mediaLLMModel, mediaLLMKey, llmTimeoutSec)
			saveConfigLocked()
			mu.Unlock()
			auditLog("CONFIG", fmt.Sprintf("Media LLM updated: model=%s url=%s timeout=%ds", req.Model, req.URL, req.Timeout))
			go nnCtrl.store(NN_SEMANTIC, "media_llm_config", fmt.Sprintf("model=%s url=%s", req.Model, req.URL))
			json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
			return
		}
		http.Error(w, "Invalid method", 405)
	})

	http.HandleFunc("/api/content_llm_config", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.Method == "GET" {
			mu.Lock()
			u, m, k, t := contentLLMURL, contentLLMModel, contentLLMKey, llmTimeoutSec
			mu.Unlock()
			json.NewEncoder(w).Encode(map[string]interface{}{"url": u, "model": m, "key": maskKey(k), "timeout": t})
			return
		}
		if r.Method == "POST" {
			var req struct {
				URL     string `json:"url"`
				Model   string `json:"model"`
				Key     string `json:"key"`
				Timeout int    `json:"timeout"`
			}
			json.NewDecoder(r.Body).Decode(&req)
			mu.Lock()
			contentLLMURL = strings.TrimRight(req.URL, "/")
			contentLLMModel = req.Model
			if req.Key != "" && !isAllAsterisks(req.Key) {
				contentLLMKey = req.Key
			}
			if req.Timeout > 0 {
				llmTimeoutSec = req.Timeout
			}
			pushLLMHistory("content", contentLLMURL, contentLLMModel, contentLLMKey, llmTimeoutSec)
			saveConfigLocked()
			mu.Unlock()
			auditLog("CONFIG", fmt.Sprintf("Content LLM updated: model=%s url=%s timeout=%ds", req.Model, req.URL, req.Timeout))
			go nnCtrl.store(NN_SEMANTIC, "content_llm_config", fmt.Sprintf("model=%s url=%s", req.Model, req.URL))
			json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
			return
		}
		http.Error(w, "Invalid method", 405)
	})

	http.HandleFunc("/api/media_llm_test", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			http.Error(w, "Invalid method", 405)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		result, err := chatWithMediaLLM("Reply with exactly one sentence confirming you are online and state your model name.")
		if err != nil {
			json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
			return
		}
		auditLog("MEDIA_LLM", "Test connection successful.")
		json.NewEncoder(w).Encode(map[string]string{"response": result})
	})

	http.HandleFunc("/api/content_llm_test", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			http.Error(w, "Invalid method", 405)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		result, err := chatWithContentLLM("Reply with exactly one sentence confirming you are online and state your model name.")
		if err != nil {
			json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
			return
		}
		auditLog("CONTENT_LLM", "Test connection successful.")
		json.NewEncoder(w).Encode(map[string]string{"response": result})
	})

	http.HandleFunc("/api/llm_history", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		slot := r.URL.Query().Get("slot")
		mu.Lock()
		hist := make([]LLMProfile, len(llmHistory[slot]))
		copy(hist, llmHistory[slot])
		mu.Unlock()
		for i := range hist {
			hist[i].Key = maskKey(hist[i].Key)
		}
		if hist == nil {
			hist = []LLMProfile{}
		}
		json.NewEncoder(w).Encode(hist)
	})

	http.HandleFunc("/api/nn/list", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(nnCtrl.listAll())
	})
	http.HandleFunc("/api/nn/spawn", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			http.Error(w, "Method not allowed", 405)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		var req struct {
			Name     string `json:"name"`
			Capacity int    `json:"capacity"`
		}
		json.NewDecoder(r.Body).Decode(&req)
		result := toolNNSpawn(map[string]interface{}{"name": req.Name, "capacity": float64(req.Capacity)})
		json.NewEncoder(w).Encode(map[string]string{"result": result})
	})
	http.HandleFunc("/api/nn/recall", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			http.Error(w, "Method not allowed", 405)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		var req struct {
			Network string `json:"network"`
			Query   string `json:"query"`
			TopK    int    `json:"top_k"`
		}
		json.NewDecoder(r.Body).Decode(&req)
		result := toolNNRecall(map[string]interface{}{"network": req.Network, "query": req.Query, "top_k": float64(req.TopK)})
		json.NewEncoder(w).Encode(map[string]string{"result": result})
	})
	http.HandleFunc("/api/nn/store", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			http.Error(w, "Method not allowed", 405)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		var req struct {
			Network string `json:"network"`
			Key     string `json:"key"`
			Value   string `json:"value"`
		}
		json.NewDecoder(r.Body).Decode(&req)
		result := toolNNStore(map[string]interface{}{"network": req.Network, "key": req.Key, "value": req.Value})
		json.NewEncoder(w).Encode(map[string]string{"result": result})
	})

	http.HandleFunc("/api/agent/spawn", spawnAgentHandler)
	http.HandleFunc("/api/agent/list", listAgentsHandler)
	http.HandleFunc("/api/agent/stop/", stopAgentHandler)
	http.HandleFunc("/api/skills/list", listSkillsHandler)


	// ─── Expert NN Chain APIs ────────────────────────────────────────────────
	http.HandleFunc("/api/nn/chain", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		var result []map[string]interface{}
		if chain.nn1 != nil {
			chain.mu.RLock()
			routeCount := len(chain.routing)
			chain.mu.RUnlock()
			result = append(result, map[string]interface{}{
				"id": "nn1_master", "name": "NN1 Master Router", "domain": "routing",
				"status": "awake", "scale_level": chain.nn1.ScaleLevel,
				"train_count": chain.nn1.TrainCount, "wake_count": chain.nn1.WakeCount,
				"route_count": routeCount, "is_nn1": true, "created_by": "seed",
			})
		}
		for _, stub := range chain.list() {
			chain.mu.RLock()
			_, isAwake := chain.awake[stub.Domain]
			chain.mu.RUnlock()
			statusStr := stub.Status
			if isAwake { statusStr = "awake" } else if statusStr != "training" { statusStr = "sleeping" }
			result = append(result, map[string]interface{}{
				"id": stub.ID, "name": stub.Name, "domain": stub.Domain,
				"status": statusStr, "scale_level": stub.ScaleLevel,
				"train_count": stub.TrainCount, "wake_count": stub.WakeCount,
				"skills": stub.SkillSet, "self_route": stub.SelfRoute,
				"last_woken": stub.LastWoken, "created_by": stub.CreatedBy,
				"is_nn1": false, "in_ram": isAwake,
			})
		}
		if result == nil { result = []map[string]interface{}{} }
		json.NewEncoder(w).Encode(result)
	})
	http.HandleFunc("/api/nn/chain/train", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" { http.Error(w, "POST required", 405); return }
		w.Header().Set("Content-Type", "application/json")
		var req struct{ Domain, Mode string }
		json.NewDecoder(r.Body).Decode(&req)
		res := toolTrainExpert(map[string]interface{}{"domain": req.Domain, "mode": req.Mode})
		json.NewEncoder(w).Encode(map[string]string{"result": res})
	})
	http.HandleFunc("/api/nn/chain/spawn", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" { http.Error(w, "POST required", 405); return }
		w.Header().Set("Content-Type", "application/json")
		var req struct {
			Name, Domain, Skills string
			Scale int
		}
		json.NewDecoder(r.Body).Decode(&req)
		res := toolSpawnExpert(map[string]interface{}{
			"name": req.Name, "domain": req.Domain,
			"skills": req.Skills, "scale": float64(req.Scale),
		})
		json.NewEncoder(w).Encode(map[string]string{"result": res})
	})
	http.HandleFunc("/api/nn/chain/routing", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		chain.mu.RLock()
		routes := chain.routing
		chain.mu.RUnlock()
		if routes == nil { routes = []RoutingEntry{} }
		if len(routes) > 300 { routes = routes[len(routes)-300:] }
		json.NewEncoder(w).Encode(routes)
	})
	http.HandleFunc("/api/nn/chain/jobs", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		trainingMu.Lock()
		jobs := trainingJobs
		trainingMu.Unlock()
		if jobs == nil { jobs = []*TrainingJob{} }
		json.NewEncoder(w).Encode(jobs)
	})

	http.HandleFunc("/api/delegation", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.Method == "GET" {
			mu.Lock()
			dc, dm, dco := delegateCoding, delegateMedia, delegateContent
			mu.Unlock()
			json.NewEncoder(w).Encode(map[string]bool{
				"coding":  dc,
				"media":   dm,
				"content": dco,
			})
			return
		}
		if r.Method == "POST" {
			var req struct {
				Coding  *bool `json:"coding"`
				Media   *bool `json:"media"`
				Content *bool `json:"content"`
			}
			json.NewDecoder(r.Body).Decode(&req)
			mu.Lock()
			if req.Coding != nil {
				delegateCoding = *req.Coding
			}
			if req.Media != nil {
				delegateMedia = *req.Media
			}
			if req.Content != nil {
				delegateContent = *req.Content
			}
			saveConfigLocked()
			mu.Unlock()
			auditLog("CONFIG", fmt.Sprintf("Delegation updated — coding:%v media:%v content:%v",
				delegateCoding, delegateMedia, delegateContent))
			json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
			return
		}
		http.Error(w, "Invalid method", 405)
	})

	http.HandleFunc("/api/websearch_config", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.Method == "GET" {
			mu.Lock()
			p, k, cx := webSearchProvider, webSearchKey, webSearchCX
			mu.Unlock()
			json.NewEncoder(w).Encode(map[string]string{"provider": p, "key": maskKey(k), "cx": cx})
			return
		}
		if r.Method == "POST" {
			var req struct {
				Provider string `json:"provider"`
				Key      string `json:"key"`
				CX       string `json:"cx"`
			}
			json.NewDecoder(r.Body).Decode(&req)
			mu.Lock()
			webSearchProvider = req.Provider
			if req.Key != "" && !isAllAsterisks(req.Key) {
				webSearchKey = req.Key
			}
			webSearchCX = req.CX
			saveConfigLocked()
			mu.Unlock()
			auditLog("CONFIG", fmt.Sprintf("Web search configured: provider=%s", req.Provider))
			json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
			return
		}
		http.Error(w, "Invalid method", 405)
	})

	http.HandleFunc("/api/websearch_test", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			http.Error(w, "Invalid method", 405)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		var req struct {
			Query string `json:"query"`
		}
		json.NewDecoder(r.Body).Decode(&req)
		if req.Query == "" {
			req.Query = "current time UTC"
		}
		result := webSearch(map[string]interface{}{"query": req.Query})
		json.NewEncoder(w).Encode(map[string]string{"result": result})
	})

	http.HandleFunc("/api/social_config", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.Method == "GET" {
			mu.Lock()
			data := map[string]interface{}{
				"discord_token":      maskKey(discordToken),
				"discord_channel":    discordChannelID,
				"twitter_key":        maskKey(twitterKey),
				"twitter_secret":     maskKey(twitterSecret),
				"twitter_token":      maskKey(twitterToken),
				"twitter_token_sec":  maskKey(twitterTokenSec),
				"telegram_token":     maskKey(telegramToken),
				"telegram_chat_id":   telegramChatID,
				"slack_token":        maskKey(slackToken),
				"slack_channel":      slackChannel,
				"whatsapp_sid":       maskKey(whatsAppSID),
				"whatsapp_token":     maskKey(whatsAppToken),
				"whatsapp_from":      whatsAppFrom,
				"whatsapp_to":        whatsAppTo,
			}
			mu.Unlock()
			json.NewEncoder(w).Encode(data)
			return
		}
		if r.Method == "POST" {
			var req struct {
				DiscordToken    string `json:"discord_token"`
				DiscordChannel  string `json:"discord_channel"`
				TwitterKey      string `json:"twitter_key"`
				TwitterSecret   string `json:"twitter_secret"`
				TwitterToken    string `json:"twitter_token"`
				TwitterTokenSec string `json:"twitter_token_sec"`
				TelegramToken   string `json:"telegram_token"`
				TelegramChatID  string `json:"telegram_chat_id"`
				SlackToken      string `json:"slack_token"`
				SlackChannel    string `json:"slack_channel"`
				WhatsAppSID     string `json:"whatsapp_sid"`
				WhatsAppToken   string `json:"whatsapp_token"`
				WhatsAppFrom    string `json:"whatsapp_from"`
				WhatsAppTo      string `json:"whatsapp_to"`
			}
			json.NewDecoder(r.Body).Decode(&req)
			mu.Lock()
			discordChannelID = req.DiscordChannel
			telegramChatID = req.TelegramChatID
			slackChannel = req.SlackChannel
			whatsAppFrom = req.WhatsAppFrom
			whatsAppTo = req.WhatsAppTo
			if req.DiscordToken != "" && !isAllAsterisks(req.DiscordToken) {
				discordToken = req.DiscordToken
			}
			if req.TwitterKey != "" && !isAllAsterisks(req.TwitterKey) {
				twitterKey = req.TwitterKey
			}
			if req.TwitterSecret != "" && !isAllAsterisks(req.TwitterSecret) {
				twitterSecret = req.TwitterSecret
			}
			if req.TwitterToken != "" && !isAllAsterisks(req.TwitterToken) {
				twitterToken = req.TwitterToken
			}
			if req.TwitterTokenSec != "" && !isAllAsterisks(req.TwitterTokenSec) {
				twitterTokenSec = req.TwitterTokenSec
			}
			if req.TelegramToken != "" && !isAllAsterisks(req.TelegramToken) {
				telegramToken = req.TelegramToken
			}
			if req.SlackToken != "" && !isAllAsterisks(req.SlackToken) {
				slackToken = req.SlackToken
			}
			if req.WhatsAppSID != "" && !isAllAsterisks(req.WhatsAppSID) {
				whatsAppSID = req.WhatsAppSID
			}
			if req.WhatsAppToken != "" && !isAllAsterisks(req.WhatsAppToken) {
				whatsAppToken = req.WhatsAppToken
			}
			saveConfigLocked()
			mu.Unlock()
			auditLog("CONFIG", "Social credentials updated.")
			json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
			return
		}
		http.Error(w, "Invalid method", 405)
	})

	http.HandleFunc("/api/social_test", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			http.Error(w, "Invalid method", 405)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		var req struct {
			Platform string `json:"platform"`
			Message  string `json:"message"`
		}
		json.NewDecoder(r.Body).Decode(&req)
		if req.Message == "" {
			req.Message = "Test message from agent."
		}
		var result string
		switch req.Platform {
		case "discord":
			result = sendDiscord(map[string]interface{}{"message": req.Message})
		case "telegram":
			result = sendTelegram(map[string]interface{}{"message": req.Message})
		case "slack":
			result = sendSlack(map[string]interface{}{"message": req.Message})
		case "twitter":
			result = sendTwitter(map[string]interface{}{"message": req.Message})
		case "whatsapp":
			result = sendWhatsApp(map[string]interface{}{"message": req.Message})
		default:
			result = "Unknown platform: " + req.Platform
		}
		ok := !strings.HasPrefix(result, "Error") && !strings.Contains(result, "not configured") && !strings.Contains(result, "error")
		json.NewEncoder(w).Encode(map[string]interface{}{"result": result, "ok": ok})
	})

	http.HandleFunc("/api/modules/search", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		q := r.URL.Query().Get("q")
		if q == "" {
			json.NewEncoder(w).Encode(map[string]interface{}{"results": []interface{}{}})
			return
		}
		results, err := searchGoPackages(q)
		if err != nil {
			json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
			return
		}
		json.NewEncoder(w).Encode(map[string]interface{}{"results": results})
	})

	http.HandleFunc("/api/modules/installed", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		mods := getInstalledModules()
		json.NewEncoder(w).Encode(map[string]interface{}{"modules": mods})
	})

	http.HandleFunc("/api/modules/add", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			http.Error(w, "Invalid method", 405)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		var req struct {
			ModulePath string `json:"module_path"`
			Version    string `json:"version"`
		}
		json.NewDecoder(r.Body).Decode(&req)
		if req.ModulePath == "" {
			json.NewEncoder(w).Encode(map[string]string{"error": "module_path required"})
			return
		}
		if req.Version == "" {
			req.Version = "latest"
		}
		result := addModule(map[string]interface{}{"module_path": req.ModulePath, "version": req.Version})
		if strings.HasPrefix(result, "go get failed") || strings.HasPrefix(result, "Error") {
			json.NewEncoder(w).Encode(map[string]string{"error": result})
			return
		}
		json.NewEncoder(w).Encode(map[string]string{"status": "ok", "output": result})
	})

	http.HandleFunc("/api/modules/remove", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			http.Error(w, "Invalid method", 405)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		var req struct {
			ModulePath string `json:"module_path"`
		}
		json.NewDecoder(r.Body).Decode(&req)
		if req.ModulePath == "" {
			json.NewEncoder(w).Encode(map[string]string{"error": "module_path required"})
			return
		}
		result := removeModule(map[string]interface{}{"module_path": req.ModulePath})
		if strings.HasPrefix(result, "go get") || strings.HasPrefix(result, "Error") {
			json.NewEncoder(w).Encode(map[string]string{"error": result})
			return
		}
		json.NewEncoder(w).Encode(map[string]string{"status": "ok", "output": result})
	})

	http.HandleFunc("/api/modules/gomod", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		content := readGoMod()
		json.NewEncoder(w).Encode(map[string]string{"content": content})
	})

	http.HandleFunc("/api/tasks/", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		pathParts := strings.Split(strings.TrimPrefix(r.URL.Path, "/api/tasks/"), "/")
		if len(pathParts) == 0 || pathParts[0] == "" {
			http.Error(w, "task id required", 400)
			return
		}
		taskID := 0
		fmt.Sscanf(pathParts[0], "%d", &taskID)
		if taskID == 0 {
			http.Error(w, "invalid task id", 400)
			return
		}

		if r.Method == "DELETE" {
			mu.Lock()
			var newTasks []Task
			found := false
			for _, t := range tasksDb {
				if t.ID == taskID {
					found = true
					continue
				}
				newTasks = append(newTasks, t)
			}
			if newTasks == nil {
				newTasks = []Task{}
			}
			tasksDb = newTasks
			saveTasksLocked()
			mu.Unlock()
			if found {
				auditLog("TASK", fmt.Sprintf("Task #%d deleted via UI", taskID))
				json.NewEncoder(w).Encode(map[string]string{"status": "deleted"})
			} else {
				http.Error(w, `{"error":"not found"}`, 404)
			}
			return
		}

		if r.Method == "PUT" || r.Method == "PATCH" {
			var req struct {
				Title       string `json:"title"`
				Description string `json:"description"`
				Status      string `json:"status"`
				Priority    int    `json:"priority"`
				Result      string `json:"result"`
			}
			json.NewDecoder(r.Body).Decode(&req)
			mu.Lock()
			updated := false
			for i, t := range tasksDb {
				if t.ID == taskID {
					if req.Title != "" {
						tasksDb[i].Title = req.Title
					}
					if req.Description != "" {
						tasksDb[i].Description = req.Description
					}
					if req.Status != "" {
						tasksDb[i].Status = req.Status
					}
					if req.Priority != 0 {
						tasksDb[i].Priority = req.Priority
					}
					if req.Result != "" {
						tasksDb[i].Result = req.Result
					}
					tasksDb[i].UpdatedAt = time.Now()
					updated = true
					break
				}
			}
			saveTasksLocked()
			mu.Unlock()
			if updated {
				auditLog("TASK", fmt.Sprintf("Task #%d updated via UI", taskID))
				json.NewEncoder(w).Encode(map[string]string{"status": "updated"})
			} else {
				http.Error(w, `{"error":"not found"}`, 404)
			}
			return
		}
		http.Error(w, "Invalid method", 405)
	})

	http.HandleFunc("/api/tasks", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.Method == "GET" {
			mu.Lock()
			tasks := make([]Task, len(tasksDb))
			copy(tasks, tasksDb)
			mu.Unlock()
			if tasks == nil {
				tasks = []Task{}
			}
			json.NewEncoder(w).Encode(tasks)
			return
		}
		if r.Method == "POST" {
			var req struct {
				Title       string `json:"title"`
				Description string `json:"description"`
				Priority    int    `json:"priority"`
			}
			json.NewDecoder(r.Body).Decode(&req)
			if req.Title == "" {
				http.Error(w, `{"error":"title required"}`, 400)
				return
			}
			if req.Priority == 0 {
				req.Priority = 2
			}
			mu.Lock()
			task := Task{
				ID:          len(tasksDb) + 1,
				Title:       req.Title,
				Description: req.Description,
				Status:      "pending",
				Priority:    req.Priority,
				CreatedAt:   time.Now(),
				UpdatedAt:   time.Now(),
			}
			tasksDb = append(tasksDb, task)
			saveTasksLocked()
			mu.Unlock()
			auditLog("TASK", fmt.Sprintf("Task #%d created via UI: %s", task.ID, task.Title))
			taskQueue <- fmt.Sprintf("TASK #%d: %s\n%s", task.ID, task.Title, task.Description)
			json.NewEncoder(w).Encode(task)
			return
		}
		http.Error(w, "Invalid method", 405)
	})

	http.HandleFunc("/api/host_config", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.Method == "GET" {
			mu.Lock()
			u, hasPass := hostUser, hostPass != ""
			mu.Unlock()
			json.NewEncoder(w).Encode(map[string]interface{}{"user": u, "pass_set": hasPass, "pass": ""})
			return
		}
		if r.Method == "POST" {
			var req struct {
				User string `json:"user"`
				Pass string `json:"pass"`
			}
			json.NewDecoder(r.Body).Decode(&req)
			mu.Lock()
			hostUser = req.User
			hostPass = req.Pass
			saveConfigLocked()
			mu.Unlock()
			if req.Pass != "" {
				auditLog("CONFIG", fmt.Sprintf("Host credentials set for user: %s", req.User))
			} else {
				auditLog("CONFIG", "Host credentials cleared.")
			}
			json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
			return
		}
		http.Error(w, "Invalid method", 405)
	})

	http.HandleFunc("/api/host_verify", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			http.Error(w, "Invalid method", 405)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		mu.Lock()
		user, pass := hostUser, hostPass
		mu.Unlock()
		if pass == "" {
			json.NewEncoder(w).Encode(map[string]string{"error": "No password configured. Set credentials first."})
			return
		}
		rootUser := user
		if rootUser == "" {
			rootUser = "root"
		}
		testCmd := fmt.Sprintf("echo %s | sudo -S -u %s bash -c 'id && whoami && echo HOST=$(hostname)'",
			shellQuote(pass), shellQuote(rootUser))
		cmd := exec.Command("bash", "-c", testCmd)
		out, err := cmd.CombinedOutput()
		if err != nil {
			json.NewEncoder(w).Encode(map[string]string{"error": fmt.Sprintf("Failed: %v\n%s", err, string(out))})
			return
		}
		auditLog("CONFIG", fmt.Sprintf("Host root access verified for user: %s", rootUser))
		json.NewEncoder(w).Encode(map[string]string{"output": strings.TrimSpace(string(out))})
	})

	http.HandleFunc("/api/ui_title", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.Method == "GET" {
			mu.Lock()
			t := webUITitle
			mu.Unlock()
			json.NewEncoder(w).Encode(map[string]string{"title": t})
			return
		}
		if r.Method == "POST" {
			var req struct {
				Title string `json:"title"`
			}
			json.NewDecoder(r.Body).Decode(&req)
			if req.Title == "" {
				req.Title = "Lily"
			}
			mu.Lock()
			webUITitle = req.Title
			saveConfigLocked()
			mu.Unlock()
			auditLog("CONFIG", fmt.Sprintf("UI title changed to: %s", req.Title))
			json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
			return
		}
		http.Error(w, "Invalid method", 405)
	})

	fmt.Printf("[System] Web Interface ready at http://localhost:%s\n", WEB_PORT)
	http.ListenAndServe(":"+WEB_PORT, nil)
}

// ─────────────────────────────────────────────────────────────────────────────
// CLI COMMAND TABLE
// ─────────────────────────────────────────────────────────────────────────────

type cliCmd struct {
	usage string
	desc  string
	run   func(args []string)
}

var cliCommands map[string]cliCmd

func initCLI() {
	cliCommands = map[string]cliCmd{
		"start": {
			"start",
			"Start the agent (default when no command given)",
			func(args []string) { runAgent() },
		},
		"task": {
			"task <text>",
			"Send a task to the running agent",
			func(args []string) {
				if len(args) == 0 {
					fmt.Println("Usage: agent task <text>")
					os.Exit(1)
				}
				cliPost("/api/task", map[string]string{"task": strings.Join(args, " ")})
				fmt.Println("✓ Task queued.")
			},
		},
		"chat": {
			"chat",
			"Show recent chat messages",
			func(args []string) {
				var msgs []map[string]interface{}
				cliGetJSON("/api/chat_messages", &msgs)
				if len(msgs) == 0 {
					fmt.Println("(no messages)")
					return
				}
				for _, m := range msgs {
					role, _ := m["role"].(string)
					content, _ := m["content"].(string)
					t, _ := m["timestamp"].(string)
					if len(t) > 16 {
						t = t[:16]
					}
					if role == "user" {
						fmt.Printf("[%s] 👤 %s\n", t, content)
					} else {
						fmt.Printf("[%s] 🤖 %s\n", t, content)
					}
				}
			},
		},
		"new-chat": {
			"new-chat",
			"Clear current chat session and start fresh",
			func(args []string) {
				cliPost("/api/new_chat", nil)
				fmt.Println("✓ New chat session started.")
			},
		},
		"logs": {
			"logs [n]",
			"Show last N audit log entries (default 20)",
			func(args []string) {
				n := 20
				if len(args) > 0 {
					fmt.Sscanf(args[0], "%d", &n)
				}
				var entries []map[string]interface{}
				cliGetJSON("/api/logs", &entries)
				if len(entries) == 0 {
					fmt.Println("(no log entries)")
					return
				}
				start := 0
				if len(entries) > n {
					start = len(entries) - n
				}
				for _, e := range entries[start:] {
					lvl, _ := e["level"].(string)
					msg, _ := e["content"].(string)
					t, _ := e["timestamp"].(string)
					if len(t) > 19 {
						t = t[:19]
					}
					fmt.Printf("[%s] %-12s %s\n", t, lvl, msg)
				}
			},
		},
		"tasks": {
			"tasks [--status pending|running|complete|failed|all]",
			"List tasks (optionally filtered by status)",
			func(args []string) {
				fs := flag.NewFlagSet("tasks", flag.ContinueOnError)
				status := fs.String("status", "all", "Filter by status")
				fs.Parse(args)
				var tasks []map[string]interface{}
				cliGetJSON("/api/tasks", &tasks)
				if len(tasks) == 0 {
					fmt.Println("(no tasks)")
					return
				}
				prio := []string{"", "🔴 HIGH", "🟡 NORMAL", "🟢 LOW"}
				for _, t := range tasks {
					st, _ := t["status"].(string)
					if *status != "all" && st != *status {
						continue
					}
					id := int(t["id"].(float64))
					title, _ := t["title"].(string)
					pri := int(t["priority"].(float64))
					pLabel := ""
					if pri >= 1 && pri <= 3 {
						pLabel = prio[pri]
					}
					result, _ := t["result"].(string)
					fmt.Printf("#%d [%-8s] %s %s\n", id, strings.ToUpper(st), title, pLabel)
					if result != "" {
						fmt.Printf("    → %s\n", result)
					}
				}
			},
		},
		"task-add": {
			"task-add --title <title> [--desc <desc>] [--priority 1|2|3]",
			"Create a new task",
			func(args []string) {
				fs := flag.NewFlagSet("task-add", flag.ContinueOnError)
				title := fs.String("title", "", "Task title (required)")
				desc := fs.String("desc", "", "Task description")
				priority := fs.Int("priority", 2, "Priority: 1=high 2=normal 3=low")
				fs.Parse(args)
				if *title == "" {
					fmt.Println("--title is required")
					os.Exit(1)
				}
				cliPost("/api/tasks", map[string]interface{}{"title": *title, "description": *desc, "priority": *priority})
				fmt.Println("✓ Task created.")
			},
		},
		"task-update": {
			"task-update <id> [--title t] [--desc d] [--status s] [--priority n] [--result r]",
			"Edit a task by ID",
			func(args []string) {
				if len(args) == 0 {
					fmt.Println("Usage: agent task-update <id> [flags]")
					os.Exit(1)
				}
				id := args[0]
				rest := args[1:]
				fs := flag.NewFlagSet("task-update", flag.ContinueOnError)
				title := fs.String("title", "", "New title")
				desc := fs.String("desc", "", "New description")
				status := fs.String("status", "", "New status: pending|running|complete|failed")
				priority := fs.Int("priority", 0, "New priority")
				result := fs.String("result", "", "New result")
				fs.Parse(rest)
				payload := map[string]interface{}{}
				if *title != "" {
					payload["title"] = *title
				}
				if *desc != "" {
					payload["description"] = *desc
				}
				if *status != "" {
					payload["status"] = *status
				}
				if *priority > 0 {
					payload["priority"] = *priority
				}
				if *result != "" {
					payload["result"] = *result
				}
				cliPut("/api/tasks/"+id, payload)
				fmt.Printf("✓ Task #%s updated.\n", id)
			},
		},
		"task-delete": {
			"task-delete <id>",
			"Delete a task by ID",
			func(args []string) {
				if len(args) == 0 {
					fmt.Println("Usage: agent task-delete <id>")
					os.Exit(1)
				}
				cliDelete("/api/tasks/" + args[0])
				fmt.Printf("✓ Task #%s deleted.\n", args[0])
			},
		},
		"memory": {
			"memory [--search query]",
			"List memories (optionally search)",
			func(args []string) {
				fs := flag.NewFlagSet("memory", flag.ContinueOnError)
				search := fs.String("search", "", "Search query")
				fs.Parse(args)
				var mems []map[string]interface{}
				cliGetJSON("/api/memory", &mems)
				q := strings.ToLower(*search)
				for _, m := range mems {
					content, _ := m["content"].(string)
					tags, _ := m["tags"].(string)
					if q != "" && !strings.Contains(strings.ToLower(content), q) && !strings.Contains(strings.ToLower(tags), q) {
						continue
					}
					fmt.Printf("[%s] %s\n", tags, content)
				}
			},
		},
		"sessions": {
			"sessions",
			"List all chat sessions",
			func(args []string) {
				var sessions []map[string]interface{}
				cliGetJSON("/api/sessions", &sessions)
				if len(sessions) == 0 {
					fmt.Println("(no sessions)")
					return
				}
				for _, s := range sessions {
					id, _ := s["id"].(float64)
					title, _ := s["title"].(string)
					t, _ := s["created_at"].(string)
					if len(t) > 19 {
						t = t[:19]
					}
					fmt.Printf("[%s] #%d — %s\n", t, int(id), title)
				}
			},
		},
		"personality": {
			"personality [--set <text>] [--show]",
			"Show or set agent personality",
			func(args []string) {
				fs := flag.NewFlagSet("personality", flag.ContinueOnError)
				set := fs.String("set", "", "New personality text")
				show := fs.Bool("show", false, "Show current personality")
				fs.Parse(args)
				if *set != "" {
					cliPost("/api/personality", map[string]string{"personality": *set})
					fmt.Println("✓ Personality updated.")
				} else {
					var d map[string]string
					cliGetJSON("/api/personality", &d)
					fmt.Println(d["personality"])
				}
				_ = show
			},
		},
		"title": {
			"title [--set <text>]",
			"Show or set the agent UI title",
			func(args []string) {
				fs := flag.NewFlagSet("title", flag.ContinueOnError)
				set := fs.String("set", "", "New title")
				fs.Parse(args)
				if *set != "" {
					cliPost("/api/ui_title", map[string]string{"title": *set})
					fmt.Println("✓ Title updated.")
				} else {
					var d map[string]string
					cliGetJSON("/api/ui_title", &d)
					fmt.Println(d["title"])
				}
			},
		},
		"llm": {
			"llm <main|coding|media|content> [--url u] [--model m] [--key k] [--timeout t]",
			"Show or configure an LLM",
			func(args []string) {
				if len(args) == 0 {
					fmt.Println("Usage: agent llm <main|coding|media|content> [flags]")
					os.Exit(1)
				}
				name := args[0]
				rest := args[1:]
				ep := map[string]string{"main": "/api/main_llm_config", "coding": "/api/coding_llm_config", "media": "/api/media_llm_config", "content": "/api/content_llm_config"}
				endpoint, ok := ep[name]
				if !ok {
					fmt.Printf("Unknown LLM: %s  (use main|coding|media|content)\n", name)
					os.Exit(1)
				}
				if len(rest) == 0 {
					var d map[string]interface{}
					cliGetJSON(endpoint, &d)
					fmt.Printf("URL:     %v\nModel:   %v\nKey:     %v\nTimeout: %v\n", d["url"], d["model"], d["key"], d["timeout"])
					return
				}
				fs := flag.NewFlagSet("llm", flag.ContinueOnError)
				url := fs.String("url", "", "Base URL")
				model := fs.String("model", "", "Model name")
				key := fs.String("key", "", "API key")
				timeout := fs.Int("timeout", 0, "Timeout seconds")
				fs.Parse(rest)
				payload := map[string]interface{}{}
				if *url != "" {
					payload["url"] = *url
				}
				if *model != "" {
					payload["model"] = *model
				}
				if *key != "" {
					payload["key"] = *key
				}
				if *timeout > 0 {
					payload["timeout"] = *timeout
				}
				cliPost(endpoint, payload)
				fmt.Printf("✓ %s LLM config updated.\n", name)
			},
		},
		"llm-test": {
			"llm-test <main|coding|media|content>",
			"Test connection to an LLM",
			func(args []string) {
				if len(args) == 0 {
					fmt.Println("Usage: agent llm-test <main|coding|media|content>")
					os.Exit(1)
				}
				ep := map[string]string{"main": "/api/main_llm_test", "coding": "/api/coding_llm_test", "media": "/api/media_llm_test", "content": "/api/content_llm_test"}
				endpoint, ok := ep[args[0]]
				if !ok {
					fmt.Printf("Unknown LLM: %s\n", args[0])
					os.Exit(1)
				}
				var d map[string]interface{}
				cliPostJSON(endpoint, nil, &d)
				if e, ok := d["error"].(string); ok {
					fmt.Printf("✗ Error: %s\n", e)
				} else {
					fmt.Printf("✓ %s\n", d["response"])
				}
			},
		},
		"delegation": {
			"delegation [--coding on|off] [--media on|off] [--content on|off]",
			"Show or toggle LLM delegation",
			func(args []string) {
				if len(args) == 0 {
					var d map[string]bool
					cliGetJSON("/api/delegation", &d)
					yn := func(b bool) string {
						if b {
							return "✓ ON"
						}
						return "✗ OFF"
					}
					fmt.Printf("Coding:  %s\nMedia:   %s\nContent: %s\n", yn(d["coding"]), yn(d["media"]), yn(d["content"]))
					return
				}
				fs := flag.NewFlagSet("delegation", flag.ContinueOnError)
				coding := fs.String("coding", "", "on|off")
				media := fs.String("media", "", "on|off")
				content := fs.String("content", "", "on|off")
				fs.Parse(args)
				payload := map[string]interface{}{}
				parseBool := func(s string) *bool {
					v := s == "on" || s == "true" || s == "1"
					return &v
				}
				if *coding != "" {
					payload["coding"] = parseBool(*coding)
				}
				if *media != "" {
					payload["media"] = parseBool(*media)
				}
				if *content != "" {
					payload["content"] = parseBool(*content)
				}
				cliPost("/api/delegation", payload)
				fmt.Println("✓ Delegation updated.")
			},
		},
		"search": {
			"search <query>",
			"Run a web search using the configured provider",
			func(args []string) {
				if len(args) == 0 {
					fmt.Println("Usage: agent search <query>")
					os.Exit(1)
				}
				query := strings.Join(args, " ")
				var d map[string]interface{}
				cliPostJSON("/api/websearch_test", map[string]string{"query": query}, &d)
				fmt.Println(d["result"])
			},
		},
		"search-config": {
			"search-config [--provider duckduckgo|google|brave] [--key k] [--cx id]",
			"Show or set web search configuration",
			func(args []string) {
				if len(args) == 0 {
					var d map[string]interface{}
					cliGetJSON("/api/websearch_config", &d)
					fmt.Printf("Provider: %v\nKey:      %v\nCX:       %v\n", d["provider"], d["key"], d["cx"])
					return
				}
				fs := flag.NewFlagSet("search-config", flag.ContinueOnError)
				provider := fs.String("provider", "", "duckduckgo|google|brave")
				key := fs.String("key", "", "API key")
				cx := fs.String("cx", "", "Google CX ID")
				fs.Parse(args)
				payload := map[string]interface{}{}
				if *provider != "" {
					payload["provider"] = *provider
				}
				if *key != "" {
					payload["key"] = *key
				}
				if *cx != "" {
					payload["cx"] = *cx
				}
				cliPost("/api/websearch_config", payload)
				fmt.Println("✓ Search config updated.")
			},
		},
		"social-send": {
			"social-send <discord|telegram|slack|twitter|whatsapp> <message>",
			"Send a message via a social platform",
			func(args []string) {
				if len(args) < 2 {
					fmt.Println("Usage: agent social-send <platform> <message>")
					os.Exit(1)
				}
				platform := args[0]
				msg := strings.Join(args[1:], " ")
				var d map[string]interface{}
				cliPostJSON("/api/social_test", map[string]string{"platform": platform, "message": msg}, &d)
				if ok, _ := d["ok"].(bool); ok {
					fmt.Printf("✓ %s\n", d["result"])
				} else {
					fmt.Printf("✗ %s\n", d["result"])
				}
			},
		},
		"social-config": {
			"social-config [--discord-token t] [--discord-channel c] [--telegram-token t] [--telegram-chat c] [--slack-token t] [--slack-channel c] [--twitter-key k] [--twitter-secret s] [--twitter-token t] [--twitter-token-sec s] [--wa-sid s] [--wa-token t] [--wa-from f] [--wa-to to]",
			"Show or set social media credentials",
			func(args []string) {
				if len(args) == 0 {
					var d map[string]interface{}
					cliGetJSON("/api/social_config", &d)
					keys := []string{"discord_token", "discord_channel", "telegram_token", "telegram_chat_id", "slack_token", "slack_channel", "twitter_key", "twitter_secret", "twitter_token", "twitter_token_sec", "whatsapp_sid", "whatsapp_token", "whatsapp_from", "whatsapp_to"}
					for _, k := range keys {
						fmt.Printf("%-22s %v\n", k+":", d[k])
					}
					return
				}
				fs := flag.NewFlagSet("social-config", flag.ContinueOnError)
				dt := fs.String("discord-token", "", "Discord bot token")
				dc := fs.String("discord-channel", "", "Discord channel ID")
				tt := fs.String("telegram-token", "", "Telegram bot token")
				tc := fs.String("telegram-chat", "", "Telegram chat ID")
				st := fs.String("slack-token", "", "Slack bot token")
				sc := fs.String("slack-channel", "", "Slack channel")
				twk := fs.String("twitter-key", "", "Twitter API key")
				tws := fs.String("twitter-secret", "", "Twitter API secret")
				twt := fs.String("twitter-token", "", "Twitter access token")
				twts := fs.String("twitter-token-sec", "", "Twitter access token secret")
				ws := fs.String("wa-sid", "", "Twilio account SID")
				wt := fs.String("wa-token", "", "Twilio auth token")
				wf := fs.String("wa-from", "", "WhatsApp from number")
				wto := fs.String("wa-to", "", "WhatsApp to number")
				fs.Parse(args)
				payload := map[string]interface{}{}
				if *dt != "" {
					payload["discord_token"] = *dt
				}
				if *dc != "" {
					payload["discord_channel"] = *dc
				}
				if *tt != "" {
					payload["telegram_token"] = *tt
				}
				if *tc != "" {
					payload["telegram_chat_id"] = *tc
				}
				if *st != "" {
					payload["slack_token"] = *st
				}
				if *sc != "" {
					payload["slack_channel"] = *sc
				}
				if *twk != "" {
					payload["twitter_key"] = *twk
				}
				if *tws != "" {
					payload["twitter_secret"] = *tws
				}
				if *twt != "" {
					payload["twitter_token"] = *twt
				}
				if *twts != "" {
					payload["twitter_token_sec"] = *twts
				}
				if *ws != "" {
					payload["whatsapp_sid"] = *ws
				}
				if *wt != "" {
					payload["whatsapp_token"] = *wt
				}
				if *wf != "" {
					payload["whatsapp_from"] = *wf
				}
				if *wto != "" {
					payload["whatsapp_to"] = *wto
				}
				cliPost("/api/social_config", payload)
				fmt.Println("✓ Social credentials updated.")
			},
		},
		"modules": {
			"modules",
			"List installed Go module dependencies",
			func(args []string) {
				var d map[string]interface{}
				cliGetJSON("/api/modules/installed", &d)
				mods, _ := d["modules"].([]interface{})
				if len(mods) == 0 {
					fmt.Println("(no modules)")
					return
				}
				for _, m := range mods {
					mm, _ := m.(map[string]interface{})
					fmt.Printf("%-50s %v\n", mm["path"], mm["version"])
				}
			},
		},
		"module-add": {
			"module-add <path> [version]",
			"Add a Go module dependency",
			func(args []string) {
				if len(args) == 0 {
					fmt.Println("Usage: agent module-add <path> [version]")
					os.Exit(1)
				}
				ver := "latest"
				if len(args) > 1 {
					ver = args[1]
				}
				var d map[string]interface{}
				cliPostJSON("/api/modules/add", map[string]string{"module_path": args[0], "version": ver}, &d)
				if e, ok := d["error"].(string); ok {
					fmt.Printf("✗ %s\n", e)
				} else {
					fmt.Printf("✓ %s\n", d["output"])
				}
			},
		},
		"module-remove": {
			"module-remove <path>",
			"Remove a Go module dependency",
			func(args []string) {
				if len(args) == 0 {
					fmt.Println("Usage: agent module-remove <path>")
					os.Exit(1)
				}
				var d map[string]interface{}
				cliPostJSON("/api/modules/remove", map[string]string{"module_path": args[0]}, &d)
				if e, ok := d["error"].(string); ok {
					fmt.Printf("✗ %s\n", e)
				} else {
					fmt.Printf("✓ %s\n", d["output"])
				}
			},
		},
		"module-search": {
			"module-search <query>",
			"Search pkg.go.dev for Go packages",
			func(args []string) {
				if len(args) == 0 {
					fmt.Println("Usage: agent module-search <query>")
					os.Exit(1)
				}
				var d map[string]interface{}
				cliGetJSONQuery("/api/modules/search", "q="+strings.Join(args, "+"), &d)
				results, _ := d["results"].([]interface{})
				if len(results) == 0 {
					fmt.Println("(no results)")
					return
				}
				for _, r := range results {
					rm, _ := r.(map[string]interface{})
					fmt.Printf("%-50s %v\n  %v\n", rm["path"], rm["version"], rm["synopsis"])
				}
			},
		},
		"gomod": {
			"gomod",
			"Show the contents of go.mod",
			func(args []string) {
				var d map[string]interface{}
				cliGetJSON("/api/modules/gomod", &d)
				fmt.Println(d["content"])
			},
		},
		"host": {
			"host [--user u] [--pass p] [--clear]",
			"Show or set host root access credentials",
			func(args []string) {
				if len(args) == 0 {
					var d map[string]interface{}
					cliGetJSON("/api/host_config", &d)
					fmt.Printf("User:   %v\nPass set: %v\n", d["user"], d["pass_set"])
					return
				}
				fs := flag.NewFlagSet("host", flag.ContinueOnError)
				user := fs.String("user", "", "Username")
				pass := fs.String("pass", "", "Password")
				clear := fs.Bool("clear", false, "Clear credentials")
				fs.Parse(args)
				if *clear {
					cliPost("/api/host_config", map[string]string{"user": "", "pass": ""})
					fmt.Println("✓ Host credentials cleared.")
					return
				}
				cliPost("/api/host_config", map[string]string{"user": *user, "pass": *pass})
				fmt.Println("✓ Host credentials updated.")
			},
		},
		"host-verify": {
			"host-verify",
			"Verify root access on the host",
			func(args []string) {
				var d map[string]interface{}
				cliPostJSON("/api/host_verify", nil, &d)
				if e, ok := d["error"].(string); ok {
					fmt.Printf("✗ %s\n", e)
				} else {
					fmt.Printf("✓ %s\n", d["output"])
				}
			},
		},
		"status": {
			"status",
			"Show agent status: LLM config, task counts, delegation, search",
			func(args []string) {
				fmt.Println("═══ Agent Status ═══")
				for _, name := range []string{"main", "coding", "media", "content"} {
					ep := "/api/" + name + "_llm_config"
					var d map[string]interface{}
					cliGetJSON(ep, &d)
					model, _ := d["model"].(string)
					url, _ := d["url"].(string)
					status := "✗ not configured"
					if url != "" {
						status = fmt.Sprintf("✓ %s @ %s", model, url)
					}
					fmt.Printf("%-10s LLM: %s\n", strings.ToUpper(name), status)
				}
				var del map[string]bool
				cliGetJSON("/api/delegation", &del)
				yn := func(b bool) string {
					if b {
						return "ON"
					}
					return "OFF"
				}
				fmt.Printf("Delegation: coding=%s  media=%s  content=%s\n", yn(del["coding"]), yn(del["media"]), yn(del["content"]))
				var tasks []map[string]interface{}
				cliGetJSON("/api/tasks", &tasks)
				counts := map[string]int{"pending": 0, "running": 0, "complete": 0, "failed": 0}
				for _, t := range tasks {
					st, _ := t["status"].(string)
					counts[st]++
				}
				fmt.Printf("Tasks: %d pending  %d running  %d complete  %d failed\n", counts["pending"], counts["running"], counts["complete"], counts["failed"])
				var sc map[string]interface{}
				cliGetJSON("/api/websearch_config", &sc)
				fmt.Printf("Search:    provider=%v\n", sc["provider"])
			},
		},
		"help": {
			"help [command]",
			"Show this help or detailed help for a command",
			func(args []string) {
				if len(args) > 0 {
					cmd, ok := cliCommands[args[0]]
					if !ok {
						fmt.Printf("Unknown command: %s\n", args[0])
						os.Exit(1)
					}
					fmt.Printf("Usage:   agent %s\n\n%s\n", cmd.usage, cmd.desc)
					return
				}
				printHelp()
			},
		},
	}
}

func printHelp() {
	fmt.Println(`
╔═══════════════════════════════════════════════════════════╗
║                  AGENT CLI COMMANDS                       ║
╚═══════════════════════════════════════════════════════════╝

AGENT CONTROL
  start                 Start the agent server (default)
  task <text>           Send a task to the running agent
  chat                  Show recent chat messages
  new-chat              Start a new chat session
  logs [n]              Show last N audit log entries (default 20)
  status                Show full agent status summary

TASK MANAGEMENT
  tasks                 List all tasks
  tasks --status <s>    Filter by: pending|running|complete|failed
  task-add              Create a new task
  task-update <id>      Edit a task by ID
  task-delete <id>      Delete a task by ID

MEMORY & SESSIONS
  memory                List stored memories
  memory --search <q>   Search memories by keyword
  sessions              List all chat sessions

PERSONALITY & TITLE
  personality           Show current personality
  personality --set <t> Set agent personality
  title                 Show UI title
  title --set <t>       Set UI title

LLM CONFIGURATION
  llm <name>            Show config for main|coding|media|content LLM
  llm <name> --url --model --key --timeout
                        Configure an LLM
  llm-test <name>       Test connection to an LLM

DELEGATION
  delegation            Show delegation status (coding/media/content)
  delegation --coding on|off --media on|off --content on|off
                        Enable or disable per-LLM delegation

WEB SEARCH
  search <query>        Run a web search
  search-config         Show search configuration
  search-config --provider --key --cx
                        Set search provider and credentials

SOCIAL MEDIA
  social-send <platform> <message>
                        Send message to discord|telegram|slack|twitter|whatsapp
  social-config         Show social credentials
  social-config [flags] Set social media credentials

GO MODULES
  modules               List installed Go dependencies
  module-add <path>     Add a Go module
  module-remove <path>  Remove a Go module
  module-search <query> Search pkg.go.dev
  gomod                 Show go.mod contents

HOST ACCESS
  host                  Show host credentials
  host --user --pass    Set host credentials
  host --clear          Clear host credentials
  host-verify           Test root access

  Run 'agent help <command>' for detailed flags on any command.
`)
}

// ─────────────────────────────────────────────────────────────────────────────
// CLI HTTP helpers
// ─────────────────────────────────────────────────────────────────────────────

func cliBase() string {
	return "http://localhost:" + WEB_PORT
}

func cliGetJSON(path string, out interface{}) {
	resp, err := http.Get(cliBase() + path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "✗ Cannot reach agent at %s — is it running?\n  %v\n", cliBase(), err)
		os.Exit(1)
	}
	defer resp.Body.Close()
	json.NewDecoder(resp.Body).Decode(out)
}

func cliGetJSONQuery(path, query string, out interface{}) {
	resp, err := http.Get(cliBase() + path + "?" + query)
	if err != nil {
		fmt.Fprintf(os.Stderr, "✗ Cannot reach agent: %v\n", err)
		os.Exit(1)
	}
	defer resp.Body.Close()
	json.NewDecoder(resp.Body).Decode(out)
}

func cliPost(path string, payload interface{}) {
	var body *bytes.Buffer
	if payload != nil {
		b, _ := json.Marshal(payload)
		body = bytes.NewBuffer(b)
	} else {
		body = bytes.NewBuffer(nil)
	}
	resp, err := http.Post(cliBase()+path, "application/json", body)
	if err != nil {
		fmt.Fprintf(os.Stderr, "✗ Cannot reach agent: %v\n", err)
		os.Exit(1)
	}
	resp.Body.Close()
}

func cliPostJSON(path string, payload interface{}, out interface{}) {
	var body *bytes.Buffer
	if payload != nil {
		b, _ := json.Marshal(payload)
		body = bytes.NewBuffer(b)
	} else {
		body = bytes.NewBuffer(nil)
	}
	resp, err := http.Post(cliBase()+path, "application/json", body)
	if err != nil {
		fmt.Fprintf(os.Stderr, "✗ Cannot reach agent: %v\n", err)
		os.Exit(1)
	}
	defer resp.Body.Close()
	json.NewDecoder(resp.Body).Decode(out)
}

func cliPut(path string, payload interface{}) {
	b, _ := json.Marshal(payload)
	req, _ := http.NewRequest("PUT", cliBase()+path, bytes.NewBuffer(b))
	req.Header.Set("Content-Type", "application/json")
	resp, err := (&http.Client{}).Do(req)
	if err != nil {
		fmt.Fprintf(os.Stderr, "✗ Cannot reach agent: %v\n", err)
		os.Exit(1)
	}
	resp.Body.Close()
}

func cliDelete(path string) {
	req, _ := http.NewRequest("DELETE", cliBase()+path, nil)
	resp, err := (&http.Client{}).Do(req)
	if err != nil {
		fmt.Fprintf(os.Stderr, "✗ Cannot reach agent: %v\n", err)
		os.Exit(1)
	}
	resp.Body.Close()
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────

func runAgent() {
	loadConfig()
	go startWebServer()
	go startHeartbeat()
	fmt.Println("[System] Agent Core Online. Heartbeat active.")
	runAgentLoop()
}

func main() {
	initCLI()

	if len(os.Args) < 2 {
		runAgent()
		return
	}

	cmdName := os.Args[1]
	cmdArgs := os.Args[2:]

	if cmdName == "-h" || cmdName == "--help" {
		printHelp()
		return
	}

	cmd, ok := cliCommands[cmdName]
	if !ok {
		fmt.Fprintf(os.Stderr, "Unknown command: %s\nRun 'agent help' for a list of commands.\n", cmdName)
		os.Exit(1)
	}
	cmd.run(cmdArgs)

	_ = sort.Search
	_ = strconv.Itoa
}

// ================= TOOL LEARNING / DELEGATOR NN =================
type ToolPattern struct {
	Task      string `json:"task"`
	Tool      string `json:"tool"`
	LLM       string `json:"llm"`
	Timestamp int64  `json:"timestamp"`
}

var TOOL_NN = "agent_tool_patterns"
var NN_MAX_CAPACITY = 5000

func learnToolPattern(task string, tool string, llm string) {
	key := task
	value := fmt.Sprintf("tool=%s llm=%s", tool, llm)

	nnCtrl.mu.Lock()
	defer nnCtrl.mu.Unlock()

	nn := nnCtrl.nets[TOOL_NN]
	if nn == nil {
		nn = &NeuralNetwork{
			Name:      TOOL_NN,
			Capacity:  NN_MAX_CAPACITY,
			W:         newWeightMatrix(),
			CreatedAt: time.Now(),
		}
		nnCtrl.nets[TOOL_NN] = nn
	}

	if nn.Stored >= nn.Capacity {
		newName := fmt.Sprintf("%s_%d", TOOL_NN, time.Now().Unix())
		auditLog("NEURAL", "Spawning new NN shard: "+newName)

		nn = &NeuralNetwork{
			Name:      newName,
			Capacity:  NN_MAX_CAPACITY,
			W:         newWeightMatrix(),
			CreatedAt: time.Now(),
		}
		nnCtrl.nets[newName] = nn
	}

	nn.Keys = append(nn.Keys, NNEntry{
		Key:      key,
		Value:    value,
		StoredAt: time.Now(),
	})

	nn.Stored++
	nn.UpdatedAt = time.Now()
}

func delegatorSelectLLM(task string) string {
	nnCtrl.mu.RLock()
	defer nnCtrl.mu.RUnlock()

	bestScore := -1.0
	bestLLM := ""

	for _, net := range nnCtrl.nets {
		if !strings.HasPrefix(net.Name, TOOL_NN) {
			continue
		}

		q := textToVec(task)

		for _, entry := range net.Keys {
			k := textToVec(entry.Key)
			score := nnDot(q, k)

			if score > bestScore {
				bestScore = score
				if strings.Contains(entry.Value, "llm=") {
					parts := strings.Split(entry.Value, "llm=")
					if len(parts) > 1 {
						bestLLM = strings.TrimSpace(parts[1])
					}
				}
			}
		}
	}

	return bestLLM
}

// ================= END TOOL LEARNING SYSTEM =================

// ================= TASK GRAPH ENGINE =================
type TaskNode struct {
	ID           string   `json:"id"`
	Description  string   `json:"description"`
	Status       string   `json:"status"`
	AgentID      string   `json:"agent_id"`
	Dependencies []string `json:"dependencies"`
	Result       string   `json:"result"`
}

type TaskGraph struct {
	Goal  string
	Nodes map[string]*TaskNode
	mu    sync.Mutex
}

var globalTaskGraph = &TaskGraph{
	Nodes: map[string]*TaskNode{},
}

func (tg *TaskGraph) AddTask(desc string, deps []string) string {
	tg.mu.Lock()
	defer tg.mu.Unlock()

	id := fmt.Sprintf("task_%d", time.Now().UnixNano())

	tg.Nodes[id] = &TaskNode{
		ID:           id,
		Description:  desc,
		Status:       "pending",
		Dependencies: deps,
	}

	return id
}

func (tg *TaskGraph) dependenciesDone(node *TaskNode) bool {
	for _, d := range node.Dependencies {
		n, ok := tg.Nodes[d]
		if !ok || n.Status != "complete" {
			return false
		}
	}
	return true
}

func (tg *TaskGraph) Run() {
	tg.mu.Lock()
	nodeIDs := make([]string, 0, len(tg.Nodes))
	for id := range tg.Nodes {
		nodeIDs = append(nodeIDs, id)
	}
	tg.mu.Unlock()

	for _, id := range nodeIDs {
		tg.mu.Lock()
		node, exists := tg.Nodes[id]
		if !exists {
			tg.mu.Unlock()
			continue
		}

		if node.Status != "pending" {
			tg.mu.Unlock()
			continue
		}

		if tg.dependenciesDone(node) {
			agent := spawnSelfAgent("worker", node.Description, "taskgraph")
			node.AgentID = agent.ID
			node.Status = "running"
			tg.mu.Unlock()

			go func(n *TaskNode) {
				result := executeAutonomousTask(n.Description)

				tg.mu.Lock()
				n.Result = result
				n.Status = "complete"
				tg.mu.Unlock()

				brain.Remember(n.Description, result)
			}(node)
		} else {
			tg.mu.Unlock()
		}
	}
}

// ================= DYNAMIC SKILL LEARNING =================
type Skill struct {
	Name        string    `json:"name"`
	Description string    `json:"description"`
	Steps       []string  `json:"steps"`
	Created     time.Time `json:"created"`
}

var skillLibrary = struct {
	mu     sync.Mutex
	skills map[string]*Skill
}{
	skills: map[string]*Skill{},
}

func learnSkill(name, desc string, steps []string) {
	skillLibrary.mu.Lock()
	defer skillLibrary.mu.Unlock()

	if skillLibrary.skills == nil {
		skillLibrary.skills = make(map[string]*Skill)
	}

	s := &Skill{
		Name:        name,
		Description: desc,
		Steps:       steps,
		Created:     time.Now(),
	}

	skillLibrary.skills[name] = s

	brain.StoreProcedural(name, strings.Join(steps, " -> "))

	auditLog("SKILL", "learned skill "+name)
}

func listSkills() []*Skill {
	skillLibrary.mu.Lock()
	defer skillLibrary.mu.Unlock()

	out := []*Skill{}

	for _, s := range skillLibrary.skills {
		out = append(out, s)
	}

	return out
}

func executeSkill(name string) string {
	skillLibrary.mu.Lock()
	s, ok := skillLibrary.skills[name]
	skillLibrary.mu.Unlock()

	if !ok {
		return "skill not found"
	}

	result := ""

	for _, step := range s.Steps {
		r := executeAutonomousTask(step)
		result += r + "\n"
	}

	return result
}

// ================= AUTONOMOUS TASK EXECUTOR =================
func executeAutonomousTask(task string) string {
	llm := delegatorSelectLLM(task)

	if llm == "" {
		llm = DEFAULT_MAIN_LLM_MODEL
	}

	req := map[string]string{
		"model":  llm,
		"prompt": task,
	}

	data, _ := json.Marshal(req)

	resp, err := http.Post(DEFAULT_MAIN_LLM_URL+"/api/generate", "application/json", bytes.NewBuffer(data))

	if err != nil {
		return err.Error()
	}

	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)

	return string(body)
}

// ================= AGENT WEB API =================
func spawnAgentHandler(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Role string `json:"role"`
		Goal string `json:"goal"`
	}

	json.NewDecoder(r.Body).Decode(&req)

	a := spawnSelfAgent(req.Role, req.Goal, "web")

	json.NewEncoder(w).Encode(a)
}

func listAgentsHandler(w http.ResponseWriter, r *http.Request) {
	agents := listSelfAgents()
	json.NewEncoder(w).Encode(agents)
}

func stopAgentHandler(w http.ResponseWriter, r *http.Request) {
	id := strings.TrimPrefix(r.URL.Path, "/api/agent/stop/")
	stopSelfAgent(id)
	w.WriteHeader(200)
}

func listSkillsHandler(w http.ResponseWriter, r *http.Request) {
	json.NewEncoder(w).Encode(listSkills())
}

// ================= END ADVANCED SYSTEMS =================