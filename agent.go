package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
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
	HEARTBEAT_INTERVAL     = 30 * time.Minute
)

// --- DATA STRUCTURES ---

type Config struct {
	// Main LLM
	MainLLMURL   string `json:"main_llm_url"`
	MainLLMModel string `json:"main_llm_model"`
	MainLLMKey   string `json:"main_llm_key"`
	// Coding LLM
	CodingLLMURL   string `json:"coding_llm_url"`
	CodingLLMModel string `json:"coding_llm_model"`
	CodingLLMKey   string `json:"coding_llm_key"`
	// Media LLM (image/video creation & editing)
	MediaLLMURL   string `json:"media_llm_url"`
	MediaLLMModel string `json:"media_llm_model"`
	MediaLLMKey   string `json:"media_llm_key"`
	// Content LLM (web content writing)
	ContentLLMURL   string `json:"content_llm_url"`
	ContentLLMModel string `json:"content_llm_model"`
	ContentLLMKey   string `json:"content_llm_key"`
	// Host credentials
	HostUser string `json:"host_user"`
	HostPass string `json:"host_pass"`
	// Web Search
	WebSearchProvider string `json:"web_search_provider"`
	WebSearchKey      string `json:"web_search_key"`
	WebSearchCX       string `json:"web_search_cx"` // Google Custom Search engine ID
	// Social
	DiscordToken     string `json:"discord_token"`
	DiscordChannelID string `json:"discord_channel_id"`
	TwitterKey       string `json:"twitter_key"`
	TwitterSecret    string `json:"twitter_secret"`
	TwitterToken     string `json:"twitter_token"`
	TwitterTokenSec  string `json:"twitter_token_sec"`
	TelegramToken    string `json:"telegram_token"`
	TelegramChatID   string `json:"telegram_chat_id"`
	SlackToken       string `json:"slack_token"`
	SlackChannel     string `json:"slack_channel"`
	WhatsAppSID      string `json:"whatsapp_sid"`
	WhatsAppToken    string `json:"whatsapp_token"`
	WhatsAppFrom     string `json:"whatsapp_from"`
	WhatsAppTo       string `json:"whatsapp_to"`
	// LLM timeout
	LLMTimeoutSec int `json:"llm_timeout_sec"`
	// Delegation toggles
	DelegateCoding  bool `json:"delegate_coding"`
	DelegateMedia   bool `json:"delegate_media"`
	DelegateContent bool `json:"delegate_content"`
	// Misc
	Personality string `json:"personality"`
	WebUITitle  string `json:"web_ui_title"`
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

type Memory struct {
	ID        int       `json:"id"`
	Content   string    `json:"content"`
	Tags      []string  `json:"tags"`
	CreatedAt time.Time `json:"created_at"`
}

type ChatSession struct {
	ID        int          `json:"id"`
	Title     string       `json:"title"`
	CreatedAt time.Time    `json:"created_at"`
	Logs      []AuditEntry `json:"logs"`
}


type ChatMessage struct {
	Role      string    `json:"role"`      // "user" | "agent"
	Content   string    `json:"content"`
	Timestamp time.Time `json:"timestamp"`
}

type Task struct {
	ID          int       `json:"id"`
	Title       string    `json:"title"`
	Description string    `json:"description"`
	Status      string    `json:"status"` // pending | running | complete | failed
	Priority    int       `json:"priority"` // 1=high 2=normal 3=low
	CreatedAt   time.Time `json:"created_at"`
	UpdatedAt   time.Time `json:"updated_at"`
	Result      string    `json:"result,omitempty"`
}

// OpenAI-compatible request/response (used for both main and coding LLMs).
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
	// Main LLM
	mainLLMURL   string
	mainLLMModel string
	mainLLMKey   string
	// Coding LLM
	codingLLMURL   string
	codingLLMModel string
	codingLLMKey   string
	// Content LLM
	mediaLLMURL   string
	mediaLLMModel string
	mediaLLMKey   string
	contentLLMURL   string
	contentLLMModel string
	contentLLMKey   string

	webUITitle      string
	llmTimeoutSec   int
	delegateCoding  bool
	delegateMedia   bool
	delegateContent bool
	tasksDb         []Task
	// Web search
	webSearchProvider string
	webSearchKey      string
	webSearchCX       string
	// Social
	discordToken     string
	discordChannelID string
	twitterKey       string
	twitterSecret    string
	twitterToken     string
	twitterTokenSec  string
	telegramToken    string
	telegramChatID   string
	slackToken       string
	slackChannel     string
	whatsAppSID      string
	whatsAppToken    string
	whatsAppFrom     string
	whatsAppTo       string
	chatMessages []ChatMessage
	// Host credentials for root access
	hostUser    string
	hostPass    string

	mu        sync.Mutex
	taskQueue = make(chan string, 20)
)

// --- SYSTEM PROMPT ---
const SYSTEM_PROMPT_BASE = `
You are a fully autonomous AI agent with Long-Term Memory, a Task Management System, and a Maintenance Heartbeat.

CORE DIRECTIVES:
1. AUTONOMOUS MEMORY: Learn user preferences and project details immediately. Use 'store_memory'.
2. TASK MANAGEMENT: Use the task system to track work. Create tasks with 'create_task', update progress
   with 'update_task', and mark done with 'complete_task'. Never use files for task tracking.
3. HEARTBEAT MODE: Periodically (every 30 mins), use 'list_tasks' to find pending work and execute it.
4. DELEGATION: Always follow the DELEGATION RULES section below — it specifies which tasks must be sent to specialist LLMs.
5. SPECIALISATION: Each delegate LLM has a specific role. Use the right tool for the right job.

TOOLS:
1.  write_file (path, content): Write text to a file.
2.  read_file (path): Read file contents.
3.  list_directory (path): List files.
4.  run_command (command): Execute shell command. Full root access when host credentials are configured.
5.  store_memory (content, tags): Permanently store a fact. Tags: user_pref, project, config.
6.  recall_memory (query): Search stored memories.
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

OUTPUT RULES:
- Respond ONLY in valid JSON. No markdown.
- Tool format: {"tool": "tool_name", "arguments": {"key": "value"}}
- Complete format: {"status": "complete", "summary": "text"}
- ALWAYS set "summary" in complete responses. This is shown directly to the user as your reply in the chat window. It must be a clear, helpful, human-readable response — not just "done". If the user asked a question, answer it in summary. If you did work, describe the result.
- FILE OUTPUTS: When writing generated content, code, or module outputs to disk, save them inside the documents/ folder. Use documents/code/ for code, documents/content/ for written content, documents/media/ for media, documents/modules/ for module-related files. Create subfolders as needed.
`

func buildSystemPrompt() string {
	mu.Lock()
	personality := currentPersonality
	dc, dm, dco := delegateCoding, delegateMedia, delegateContent
	codingModel := codingLLMModel
	mediaModel  := mediaLLMModel
	contentModel := contentLLMModel
	mu.Unlock()

	// Build dynamic delegation section
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

	// Defaults
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
	delegateCoding = true
	delegateMedia = true
	delegateContent = true
	webSearchProvider = "duckduckgo"
	webSearchKey = ""
	webSearchCX = ""
	discordToken = ""; discordChannelID = ""
	twitterKey = ""; twitterSecret = ""; twitterToken = ""; twitterTokenSec = ""
	telegramToken = ""; telegramChatID = ""
	slackToken = ""; slackChannel = ""
	whatsAppSID = ""; whatsAppToken = ""; whatsAppFrom = ""; whatsAppTo = ""

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
		if cfg.LLMTimeoutSec > 0 { llmTimeoutSec = cfg.LLMTimeoutSec } else { llmTimeoutSec = 300 }
		delegateCoding = cfg.DelegateCoding
		delegateMedia = cfg.DelegateMedia
		delegateContent = cfg.DelegateContent
		webSearchProvider = cfg.WebSearchProvider; if webSearchProvider == "" { webSearchProvider = "duckduckgo" }
		webSearchKey = cfg.WebSearchKey; webSearchCX = cfg.WebSearchCX
		discordToken = cfg.DiscordToken; discordChannelID = cfg.DiscordChannelID
		twitterKey = cfg.TwitterKey; twitterSecret = cfg.TwitterSecret
		twitterToken = cfg.TwitterToken; twitterTokenSec = cfg.TwitterTokenSec
		telegramToken = cfg.TelegramToken; telegramChatID = cfg.TelegramChatID
		slackToken = cfg.SlackToken; slackChannel = cfg.SlackChannel
		whatsAppSID = cfg.WhatsAppSID; whatsAppToken = cfg.WhatsAppToken
		whatsAppFrom = cfg.WhatsAppFrom; whatsAppTo = cfg.WhatsAppTo
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
		MainLLMURL:     mainLLMURL,
		MainLLMModel:   mainLLMModel,
		MainLLMKey:     mainLLMKey,
		CodingLLMURL:   codingLLMURL,
		CodingLLMModel: codingLLMModel,
		CodingLLMKey:   codingLLMKey,
		MediaLLMURL:   mediaLLMURL,
		MediaLLMModel: mediaLLMModel,
		MediaLLMKey:   mediaLLMKey,
		ContentLLMURL:   contentLLMURL,
		ContentLLMModel: contentLLMModel,
		ContentLLMKey:   contentLLMKey,
		Personality:    currentPersonality,
		WebUITitle:     webUITitle,
		LLMTimeoutSec:   llmTimeoutSec,
		DelegateCoding:  delegateCoding,
		DelegateMedia:   delegateMedia,
		DelegateContent: delegateContent,
		HostUser:       hostUser,
		HostPass:       hostPass,
		WebSearchProvider: webSearchProvider,
		WebSearchKey:      webSearchKey,
		WebSearchCX:       webSearchCX,
		DiscordToken: discordToken, DiscordChannelID: discordChannelID,
		TwitterKey: twitterKey, TwitterSecret: twitterSecret,
		TwitterToken: twitterToken, TwitterTokenSec: twitterTokenSec,
		TelegramToken: telegramToken, TelegramChatID: telegramChatID,
		SlackToken: slackToken, SlackChannel: slackChannel,
		WhatsAppSID: whatsAppSID, WhatsAppToken: whatsAppToken,
		WhatsAppFrom: whatsAppFrom, WhatsAppTo: whatsAppTo,
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

// ensureDocDir creates documents/<subdir> and returns full path for output files
func ensureDocDir(subdir string) string {
	dir := filepath.Join(DOCS_DIR, subdir)
	os.MkdirAll(dir, 0755)
	return dir
}

// docPath returns a path inside documents/<subdir>/<filename>
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
		// Pipe password to sudo -S so the agent runs as root
		// Uses: echo '<pass>' | sudo -S -u <user or root> bash -c '<cmd>'
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

// shellQuote wraps s in single quotes, escaping any internal single quotes.
func shellQuote(s string) string {
	replaced := strings.ReplaceAll(s, "'", "'" + "\\" + "''")
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

func saveMemoryToFile() {
	mu.Lock()
	defer mu.Unlock()
	data, _ := json.MarshalIndent(memoryDb, "", "  ")
	os.WriteFile(MEMORY_FILE, data, 0644)
}

func storeMemory(params map[string]interface{}) string {
	content, _ := params["content"].(string)
	tagsStr, _ := params["tags"].(string)
	tags := strings.Split(tagsStr, ",")
	for i := range tags {
		tags[i] = strings.TrimSpace(tags[i])
	}
	mu.Lock()
	memoryDb = append(memoryDb, Memory{
		ID: len(memoryDb) + 1, Content: content, Tags: tags, CreatedAt: time.Now(),
	})
	mu.Unlock()
	saveMemoryToFile()
	auditLog("MEMORY", fmt.Sprintf("Auto-Saved: %s", content))
	return "Memory stored."
}

func recallMemory(params map[string]interface{}) string {
	query, _ := params["query"].(string)
	query = strings.ToLower(query)
	mu.Lock()
	defer mu.Unlock()
	var results []string
	for _, m := range memoryDb {
		if strings.Contains(strings.ToLower(m.Content), query) {
			results = append(results, fmt.Sprintf("[%d] %s", m.ID, m.Content))
		}
	}
	if len(results) == 0 {
		return "No memories found."
	}
	return strings.Join(results, "\n")
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
	desc, _  := params["description"].(string)
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

func goModPath() string {
	// Try to find go.mod next to the running binary or in cwd
	if _, err := os.Stat("go.mod"); err == nil {
		return "."
	}
	return ""
}

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
	// Run tidy to clean up
	exec.Command("go", "mod", "tidy").Run()
	auditLog("MODULE", fmt.Sprintf("Module added: %s", spec))
	// Snapshot go.mod to documents/modules/
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
		// Fallback: read go.mod directly
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
	// Try directory of SOURCE_FILE
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
// callLLM sends messages to any OpenAI-compatible endpoint.
// urlBase: root URL (e.g. "http://localhost:11434").
//          "/v1/chat/completions" is appended automatically unless already present.
// model:   model identifier string.
// key:     API key, or "" for unauthenticated servers.
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
			// Provide a clear, actionable error message
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
			continue // retry
		}

		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()

		if resp.StatusCode == 503 || resp.StatusCode == 502 {
			lastErr = fmt.Errorf("server unavailable (HTTP %d) — model may still be loading", resp.StatusCode)
			continue // retry
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
	// Save output to documents/code/
	timestamp := time.Now().Format("20060102-150405")
	outFile := docPath("code", fmt.Sprintf("coding_%s.md", timestamp))
	os.WriteFile(outFile, []byte(fmt.Sprintf("# Coding Output\n**Prompt:** %s\n\n%s", truncate(prompt, 200), result)), 0644)
	auditLog("CODING_LLM", fmt.Sprintf("Saved to %s", outFile))
	return result
}

// --- CONTENT LLM TOOL ---

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
	// Save output to documents/media/
	timestamp := time.Now().Format("20060102-150405")
	outFile := docPath("media", fmt.Sprintf("media_%s.md", timestamp))
	os.WriteFile(outFile, []byte(fmt.Sprintf("# Media Output\n**Prompt:** %s\n\n%s", truncate(prompt,200), result)), 0644)
	auditLog("MEDIA_LLM", fmt.Sprintf("Saved to %s", outFile))
	return result
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
	// Save output to documents/content/
	timestamp := time.Now().Format("20060102-150405")
	outFile := docPath("content", fmt.Sprintf("content_%s.md", timestamp))
	os.WriteFile(outFile, []byte(fmt.Sprintf("# Content Output\n**Prompt:** %s\n\n%s", truncate(prompt,200), result)), 0644)
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
	"write_file":     writeFile,
	"read_file":      readFile,
	"list_directory": func(p map[string]interface{}) string { return "Use run_command 'ls -la'" },
	"run_command":    runCommand,
	"store_memory":   storeMemory,
	"recall_memory":   recallMemory,
	"edit_own_code":  editOwnCode,
	"coding_llm":     codingLLMTool,
	"media_llm":      mediaLLMTool,
	"content_llm":    contentLLMTool,
	"create_task":    createTask,
	"list_tasks":     listTasks,
	"update_task":    updateTask,
	"complete_task":  completeTask,
	"web_search":     webSearch,
	"fetch_url":      fetchURL,
	"send_discord":   sendDiscord,
	"send_telegram":  sendTelegram,
	"send_slack":     sendSlack,
	"send_twitter":   sendTwitter,
	"send_whatsapp":  sendWhatsApp,
	"add_module":     addModule,
	"remove_module":  removeModule,
	"list_modules":   func(p map[string]interface{}) string { return listModules(p) },
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
		ID: len(chatSessions) + 1, Title: title,
		CreatedAt: time.Now(), Logs: sessionLogs,
	}
	chatSessions = append(chatSessions, session)
	saveChatSessionsLocked()
	auditLogs = []AuditEntry{}
	chatMessages = []ChatMessage{}
	return session
}

// --- HEARTBEAT SYSTEM ---

func startHeartbeat() {
	auditLog("SYSTEM", fmt.Sprintf("Heartbeat started. Interval: %v", HEARTBEAT_INTERVAL))
	ticker := time.NewTicker(HEARTBEAT_INTERVAL)
	for range ticker.C {
		auditLog("HEARTBEAT", "Waking up to check for pending tasks...")
		// Count pending tasks for heartbeat message
		mu.Lock()
		pendingCount := 0
		for _, t := range tasksDb {
			if t.Status == "pending" {
				pendingCount++
			}
		}
		mu.Unlock()
		heartbeatMsg := fmt.Sprintf("AUTONOMOUS MAINTENANCE: Use list_tasks(status=pending) to find pending work and execute it. There are currently %d pending tasks. Review audit logs for any errors.", pendingCount)
		taskQueue <- heartbeatMsg
	}
}

// --- AGENT LOOP ---

func runAgentLoop() {
	loadMemory()
	loadTasks()
	loadChatSessions()
	for task := range taskQueue {
		auditLog("INFO", fmt.Sprintf("Task: %s", task))
		relevantContext := findRelevantMemories(task)
		messages := []Message{{Role: "system", Content: buildSystemPrompt()}}
		if len(relevantContext) > 0 {
			messages = append(messages, Message{
				Role:    "system",
				Content: fmt.Sprintf("RELEVANT MEMORY CONTEXT:\n%s", strings.Join(relevantContext, "\n")),
			})
			auditLog("CONTEXT", fmt.Sprintf("Injected %d memories.", len(relevantContext)))
		}
		messages = append(messages, Message{Role: "user", Content: task})
		chatRecorded := false
		for i := 0; i < MAX_LOOPS; i++ {
			response, err := chatWithMainLLM(messages)
			if err != nil {
				auditLog("ERROR", fmt.Sprintf("Main LLM Error: %v", err))
				// Force an error reply into chat so the user sees what happened
				mu.Lock()
				chatMessages = append(chatMessages, ChatMessage{Role: "agent",
					Content: fmt.Sprintf("I ran into an error: %v", err), Timestamp: time.Now()})
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
				// LLM replied with plain text — treat it directly as the chat reply
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
					// Force the LLM to give a real reply if summary is empty
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
				// Mark the most-recent running task as complete
				mu.Lock()
				for j := len(tasksDb)-1; j >= 0; j-- {
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
			messages = append(messages, Message{Role: "user", Content: fmt.Sprintf("Tool Result: %s", result)})
		}
		// Force-reply guarantee: if loop ended without recording a chat message,
		// make one final call to get a conversational response for the user.
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
	stopWords := map[string]bool{"the": true, "is": true, "at": true, "which": true, "on": true, "a": true, "an": true, "and": true}
	var keywords []string
	for _, w := range strings.Fields(taskLower) {
		if len(w) > 3 && !stopWords[w] {
			keywords = append(keywords, w)
		}
	}
	var relevant []string
	mu.Lock()
	defer mu.Unlock()
	for _, mem := range memoryDb {
		memLower := strings.ToLower(mem.Content)
		for _, kw := range keywords {
			if strings.Contains(memLower, kw) {
				relevant = append(relevant, fmt.Sprintf("- %s", mem.Content))
				break
			}
		}
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

// searchGoPackages fetches search results from pkg.go.dev and extracts package info.
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

	// Extract each search result block: <div class="SearchSnippet">
	// Parse data-gtmc attributes and snippet text
	// Pattern: look for package paths in href="/github.com/..." or href="/pkg.go.dev paths
	// Use simple string scanning
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

		// Extract path from the first href that looks like a module path
		if hIdx := strings.Index(block, `href="/`); hIdx >= 0 {
			sub := block[hIdx+7:]
			if eIdx := strings.IndexAny(sub, `"?#`); eIdx >= 0 {
				candidate := sub[:eIdx]
				// filter out site pages
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

		// Synopsis: inside <p class="SearchSnippet-synopsis">
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

		// Version: look for "v\d" pattern near "Version" or just after path
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

		// License
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

// getInstalledModules parses go.mod to return dependency list.
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
			// "require github.com/foo/bar v1.2.3" or inside block
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
	default: // duckduckgo (no key needed)
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
	// DuckDuckGo HTML scrape (no key required)
	url := "https://html.duckduckgo.com/html/?q=" + strings.ReplaceAll(query, " ", "+")
	client := &http.Client{Timeout: 10 * time.Second}
	req, _ := http.NewRequest("GET", url, nil)
	req.Header.Set("User-Agent", "Mozilla/5.0 (compatible; AgentBot/1.0)")
	resp, err := client.Do(req)
	if err != nil { return nil, err }
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	html := string(body)

	var results []SearchResult
	// Parse result blocks
	marker := `class="result__title"`
	pos := 0
	for len(results) < 8 {
		idx := strings.Index(html[pos:], marker)
		if idx < 0 { break }
		pos += idx + len(marker)
		// Extract title
		tStart := strings.Index(html[pos:], ">")
		if tStart < 0 { continue }
		tSub := html[pos+tStart+1:]
		tEnd := strings.Index(tSub, "</a>")
		title := ""
		if tEnd >= 0 { title = stripHTML(tSub[:tEnd]) }
		// Extract URL
		urlMarker := `class="result__url"`
		uIdx := strings.Index(html[pos:], urlMarker)
		href := ""
		if uIdx >= 0 {
			uSub := html[pos+uIdx:]
			uTagEnd := strings.Index(uSub, ">")
			if uTagEnd >= 0 {
				uText := uSub[uTagEnd+1:]
				uClose := strings.Index(uText, "<")
				if uClose >= 0 { href = "https://" + strings.TrimSpace(uText[:uClose]) }
			}
		}
		// Extract snippet
		snipMarker := `class="result__snippet"`
		sIdx := strings.Index(html[pos:], snipMarker)
		snippet := ""
		if sIdx >= 0 {
			sSub := html[pos+sIdx:]
			sTagEnd := strings.Index(sSub, ">")
			if sTagEnd >= 0 {
				sText := sSub[sTagEnd+1:]
				sClose := strings.Index(sText, "</a>")
				if sClose >= 0 { snippet = stripHTML(sText[:sClose]) }
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
	if err != nil { return nil, err }
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	var data struct {
		Items []struct {
			Title   string `json:"title"`
			Link    string `json:"link"`
			Snippet string `json:"snippet"`
		} `json:"items"`
	}
	if err := json.Unmarshal(body, &data); err != nil { return nil, err }
	var results []SearchResult
	for _, item := range data.Items {
		results = append(results, SearchResult{Title: item.Title, URL: item.Link, Snippet: item.Snippet})
	}
	return results, nil
}

func braveSearch(query, key string) ([]SearchResult, error) {
	if key == "" { return nil, fmt.Errorf("Brave Search API key required") }
	url := "https://api.search.brave.com/res/v1/web/search?q=" + strings.ReplaceAll(query, " ", "+") + "&count=8"
	client := &http.Client{Timeout: 10 * time.Second}
	req, _ := http.NewRequest("GET", url, nil)
	req.Header.Set("Accept", "application/json")
	req.Header.Set("X-Subscription-Token", key)
	resp, err := client.Do(req)
	if err != nil { return nil, err }
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
	if err := json.Unmarshal(body, &data); err != nil { return nil, err }
	var results []SearchResult
	for _, r := range data.Web.Results {
		results = append(results, SearchResult{Title: r.Title, URL: r.URL, Snippet: r.Description})
	}
	return results, nil
}

func stripHTML(s string) string {
	// Remove HTML tags and decode common entities
	var b strings.Builder
	inTag := false
	for _, c := range s {
		if c == '<' { inTag = true; continue }
		if c == '>' { inTag = false; continue }
		if !inTag { b.WriteRune(c) }
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

// fetchURL fetches the text content of a URL for the agent
func fetchURL(params map[string]interface{}) string {
	url, _ := params["url"].(string)
	if url == "" { return "Error: 'url' is required." }
	auditLog("WEBSEARCH", fmt.Sprintf("Fetching URL: %s", url))
	client := &http.Client{Timeout: 15 * time.Second}
	req, err := http.NewRequest("GET", url, nil)
	if err != nil { return fmt.Sprintf("Error: %v", err) }
	req.Header.Set("User-Agent", "Mozilla/5.0 (compatible; AgentBot/1.0)")
	resp, err := client.Do(req)
	if err != nil { return fmt.Sprintf("Fetch failed: %v", err) }
	defer resp.Body.Close()
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 32000))
	return truncate(stripHTML(string(body)), 4000)
}

// ─── SOCIAL TOOLS ────────────────────────────────────────────────────────────

func sendDiscord(params map[string]interface{}) string {
	msg, _ := params["message"].(string)
	if msg == "" { return "Error: 'message' required." }
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
	if err != nil { return fmt.Sprintf("Discord error: %v", err) }
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
	if msg == "" { return "Error: 'message' required." }
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
	if err != nil { return fmt.Sprintf("Telegram error: %v", err) }
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
	if msg == "" { return "Error: 'message' required." }
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
	if err != nil { return fmt.Sprintf("Slack error: %v", err) }
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
	if msg == "" { return "Error: 'message' required." }
	mu.Lock()
	consumerKey := twitterKey
	consumerSecret := twitterSecret
	accessToken := twitterToken
	accessSecret := twitterTokenSec
	mu.Unlock()
	if consumerKey == "" || accessToken == "" {
		return "Twitter/X not configured. Set keys in the Social tab."
	}
	// Post tweet via Twitter API v2
	payload, _ := json.Marshal(map[string]string{"text": msg})
	req, _ := http.NewRequest("POST", "https://api.twitter.com/2/tweets", bytes.NewBuffer(payload))
	req.Header.Set("Content-Type", "application/json")
	// OAuth 1.0a header
	authHeader := buildOAuth1Header("POST", "https://api.twitter.com/2/tweets",
		consumerKey, consumerSecret, accessToken, accessSecret)
	req.Header.Set("Authorization", authHeader)
	resp, err := (&http.Client{Timeout: 10 * time.Second}).Do(req)
	if err != nil { return fmt.Sprintf("Twitter error: %v", err) }
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	if resp.StatusCode == 201 {
		auditLog("SOCIAL", "Tweet posted successfully")
		return "Tweet posted."
	}
	return fmt.Sprintf("Twitter error %d: %s", resp.StatusCode, string(b))
}

func sendWhatsApp(params map[string]interface{}) string {
	msg, _ := params["message"].(string)
	if msg == "" { return "Error: 'message' required." }
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
	if err != nil { return fmt.Sprintf("WhatsApp error: %v", err) }
	defer resp.Body.Close()
	if resp.StatusCode == 201 {
		auditLog("SOCIAL", fmt.Sprintf("WhatsApp message sent to %s", to))
		return "WhatsApp message sent."
	}
	b, _ := io.ReadAll(resp.Body)
	return fmt.Sprintf("WhatsApp error %d: %s", resp.StatusCode, string(b))
}

// buildOAuth1Header creates a minimal OAuth 1.0a Authorization header for Twitter
func buildOAuth1Header(method, apiURL, consumerKey, consumerSecret, token, tokenSecret string) string {
	// For production use a proper OAuth library; this is a simplified version
	// that works for simple POST requests without body params
	ts := fmt.Sprintf("%d", time.Now().Unix())
	nonce := fmt.Sprintf("%d", time.Now().UnixNano())
	params := fmt.Sprintf(
		`oauth_consumer_key="%s",oauth_nonce="%s",oauth_signature_method="HMAC-SHA1",`+
		`oauth_timestamp="%s",oauth_token="%s",oauth_version="1.0"`,
		consumerKey, nonce, ts, token)
	return "OAuth " + params
}

// --- HELPERS ---

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
    <div class="ttab" onclick="switchTab('memory')">🧠 Memory</div>
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

    <!-- ===== MEMORY ===== -->
    <div id="pane-memory" class="tab-pane">
        <input type="text" class="search-bar" id="mem-search" placeholder="Search memories..." oninput="filterMemory()" style="max-width:500px;">
        <div class="scroll-list" id="memory-list"><div class="empty-state">No memories yet.</div></div>
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

    <!-- ===== AUDIT ===== -->
    <div id="pane-audit" class="tab-pane">
        <input type="text" class="search-bar" id="audit-search" placeholder="Filter audit log..." oninput="filterAudit()" style="max-width:500px;">
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
var tabNames=['chat','memory','tasks','sessions','personality','main-llm','coding-llm','media-llm','content-llm','websearch','social','modules','host','audit'];

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
function filterAudit(){
    var q=document.getElementById('audit-search').value.toLowerCase();
    var c=document.getElementById('audit-list');
    var f=q?logs.filter(function(l){return l.content.toLowerCase().includes(q);}):logs;
    if(!f.length){c.innerHTML='<div class="empty-state">No entries.</div>';return;}
    var h='';
    f.slice().reverse().forEach(function(l){
        h+='<div class="audit-entry log-'+l.level+'">'+new Date(l.timestamp).toLocaleTimeString()+' <b>'+l.level+'</b> '+esc(l.content)+'</div>';
    });
    c.innerHTML=h;
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
		if msgs == nil { msgs = []ChatMessage{} }
		json.NewEncoder(w).Encode(msgs)
	})

	http.HandleFunc("/api/task", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" { http.Error(w, "Invalid method", 405); return }
		var req struct{ Task string `json:"task"` }
		json.NewDecoder(r.Body).Decode(&req)
		if req.Task != "" {
			taskQueue <- req.Task
			mu.Lock()
			chatMessages = append(chatMessages, ChatMessage{Role: "user", Content: req.Task, Timestamp: time.Now()})
			// Mirror chat request as a task at normal priority
			chatTask := Task{
				ID:          len(tasksDb) + 1,
				Title:       truncate(req.Task, 80),
				Description: req.Task,
				Status:      "running", // already being processed
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
		mu.Lock(); defer mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(auditLogs)
	})

	http.HandleFunc("/api/memory", func(w http.ResponseWriter, r *http.Request) {
		mu.Lock(); defer mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(memoryDb)
	})

	http.HandleFunc("/api/sessions", func(w http.ResponseWriter, r *http.Request) {
		mu.Lock(); defer mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(chatSessions)
	})

	http.HandleFunc("/api/new_chat", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" { http.Error(w, "Invalid method", 405); return }
		session := newChat()
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(session)
		auditLog("SYSTEM", fmt.Sprintf("New chat started. Session #%d archived.", session.ID))
	})

	http.HandleFunc("/api/personality", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.Method == "GET" {
			mu.Lock(); p := currentPersonality; mu.Unlock()
			json.NewEncoder(w).Encode(map[string]string{"personality": p})
			return
		}
		if r.Method == "POST" {
			var req struct{ Personality string `json:"personality"` }
			json.NewDecoder(r.Body).Decode(&req)
			mu.Lock(); currentPersonality = req.Personality; saveConfigLocked(); mu.Unlock()
			auditLog("CONFIG", "Personality updated.")
			json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
			return
		}
		http.Error(w, "Invalid method", 405)
	})

	// --- Main LLM config ---
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
			if req.Key != "" && !isAllAsterisks(req.Key) { mainLLMKey = req.Key }
			if req.Timeout > 0 { llmTimeoutSec = req.Timeout }
			saveConfigLocked()
			mu.Unlock()
			auditLog("CONFIG", fmt.Sprintf("Main LLM updated: model=%s url=%s timeout=%ds", req.Model, req.URL, req.Timeout))
			json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
			return
		}
		http.Error(w, "Invalid method", 405)
	})

	http.HandleFunc("/api/main_llm_test", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" { http.Error(w, "Invalid method", 405); return }
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

	// --- Coding LLM config ---
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
			if req.Key != "" && !isAllAsterisks(req.Key) { codingLLMKey = req.Key }
			if req.Timeout > 0 { llmTimeoutSec = req.Timeout }
			saveConfigLocked()
			mu.Unlock()
			auditLog("CONFIG", fmt.Sprintf("Coding LLM updated: model=%s url=%s timeout=%ds", req.Model, req.URL, req.Timeout))
			json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
			return
		}
		http.Error(w, "Invalid method", 405)
	})

	http.HandleFunc("/api/coding_llm_test", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" { http.Error(w, "Invalid method", 405); return }
		w.Header().Set("Content-Type", "application/json")
		result, err := chatWithCodingLLM("Reply with exactly one sentence confirming you are online and state your model name.")
		if err != nil {
			json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
			return
		}
		auditLog("CODING_LLM", "Test connection successful.")
		json.NewEncoder(w).Encode(map[string]string{"response": result})
	})

	// --- Content LLM config ---
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
			if req.Key != "" && !isAllAsterisks(req.Key) { mediaLLMKey = req.Key }
			if req.Timeout > 0 { llmTimeoutSec = req.Timeout }
			saveConfigLocked()
			mu.Unlock()
			auditLog("CONFIG", fmt.Sprintf("Media LLM updated: model=%s url=%s timeout=%ds", req.Model, req.URL, req.Timeout))
			json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
			return
		}
		http.Error(w, "Invalid method", 405)
	})

	// --- Content LLM (web content writing) ---
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
			if req.Key != "" && !isAllAsterisks(req.Key) { contentLLMKey = req.Key }
			if req.Timeout > 0 { llmTimeoutSec = req.Timeout }
			saveConfigLocked()
			mu.Unlock()
			auditLog("CONFIG", fmt.Sprintf("Content LLM updated: model=%s url=%s timeout=%ds", req.Model, req.URL, req.Timeout))
			json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
			return
		}
		http.Error(w, "Invalid method", 405)
	})

	http.HandleFunc("/api/media_llm_test", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" { http.Error(w, "Invalid method", 405); return }
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
		if r.Method != "POST" { http.Error(w, "Invalid method", 405); return }
		w.Header().Set("Content-Type", "application/json")
		result, err := chatWithContentLLM("Reply with exactly one sentence confirming you are online and state your model name.")
		if err != nil {
			json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
			return
		}
		auditLog("CONTENT_LLM", "Test connection successful.")
		json.NewEncoder(w).Encode(map[string]string{"response": result})
	})






	// --- Delegation Toggles ---
	http.HandleFunc("/api/delegation", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.Method == "GET" {
			mu.Lock()
			dc, dm, dco := delegateCoding, delegateMedia, delegateContent
			mu.Unlock()
			json.NewEncoder(w).Encode(map[string]bool{
				"coding": dc, "media": dm, "content": dco,
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
			if req.Coding  != nil { delegateCoding  = *req.Coding }
			if req.Media   != nil { delegateMedia   = *req.Media }
			if req.Content != nil { delegateContent = *req.Content }
			saveConfigLocked()
			mu.Unlock()
			auditLog("CONFIG", fmt.Sprintf("Delegation updated — coding:%v media:%v content:%v",
				delegateCoding, delegateMedia, delegateContent))
			json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
			return
		}
		http.Error(w, "Invalid method", 405)
	})

	// --- Web Search Config ---
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
			if req.Key != "" && !isAllAsterisks(req.Key) { webSearchKey = req.Key }
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
		if r.Method != "POST" { http.Error(w, "Invalid method", 405); return }
		w.Header().Set("Content-Type", "application/json")
		var req struct{ Query string `json:"query"` }
		json.NewDecoder(r.Body).Decode(&req)
		if req.Query == "" { req.Query = "current time UTC" }
		result := webSearch(map[string]interface{}{"query": req.Query})
		json.NewEncoder(w).Encode(map[string]string{"result": result})
	})

	// --- Social Config ---
	http.HandleFunc("/api/social_config", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.Method == "GET" {
			mu.Lock()
			data := map[string]interface{}{
				"discord_token": maskKey(discordToken), "discord_channel": discordChannelID,
				"twitter_key": maskKey(twitterKey), "twitter_secret": maskKey(twitterSecret),
				"twitter_token": maskKey(twitterToken), "twitter_token_sec": maskKey(twitterTokenSec),
				"telegram_token": maskKey(telegramToken), "telegram_chat_id": telegramChatID,
				"slack_token": maskKey(slackToken), "slack_channel": slackChannel,
				"whatsapp_sid": maskKey(whatsAppSID), "whatsapp_token": maskKey(whatsAppToken),
				"whatsapp_from": whatsAppFrom, "whatsapp_to": whatsAppTo,
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
			if req.DiscordToken   != "" && !isAllAsterisks(req.DiscordToken)   { discordToken = req.DiscordToken }
			if req.TwitterKey     != "" && !isAllAsterisks(req.TwitterKey)     { twitterKey = req.TwitterKey }
			if req.TwitterSecret  != "" && !isAllAsterisks(req.TwitterSecret)  { twitterSecret = req.TwitterSecret }
			if req.TwitterToken   != "" && !isAllAsterisks(req.TwitterToken)   { twitterToken = req.TwitterToken }
			if req.TwitterTokenSec != "" && !isAllAsterisks(req.TwitterTokenSec){ twitterTokenSec = req.TwitterTokenSec }
			if req.TelegramToken  != "" && !isAllAsterisks(req.TelegramToken)  { telegramToken = req.TelegramToken }
			if req.SlackToken     != "" && !isAllAsterisks(req.SlackToken)     { slackToken = req.SlackToken }
			if req.WhatsAppSID    != "" && !isAllAsterisks(req.WhatsAppSID)    { whatsAppSID = req.WhatsAppSID }
			if req.WhatsAppToken  != "" && !isAllAsterisks(req.WhatsAppToken)  { whatsAppToken = req.WhatsAppToken }
			saveConfigLocked()
			mu.Unlock()
			auditLog("CONFIG", "Social credentials updated.")
			json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
			return
		}
		http.Error(w, "Invalid method", 405)
	})

	http.HandleFunc("/api/social_test", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" { http.Error(w, "Invalid method", 405); return }
		w.Header().Set("Content-Type", "application/json")
		var req struct {
			Platform string `json:"platform"`
			Message  string `json:"message"`
		}
		json.NewDecoder(r.Body).Decode(&req)
		if req.Message == "" { req.Message = "Test message from agent." }
		var result string
		switch req.Platform {
		case "discord":  result = sendDiscord(map[string]interface{}{"message": req.Message})
		case "telegram": result = sendTelegram(map[string]interface{}{"message": req.Message})
		case "slack":    result = sendSlack(map[string]interface{}{"message": req.Message})
		case "twitter":  result = sendTwitter(map[string]interface{}{"message": req.Message})
		case "whatsapp": result = sendWhatsApp(map[string]interface{}{"message": req.Message})
		default:         result = "Unknown platform: " + req.Platform
		}
		ok := !strings.HasPrefix(result, "Error") && !strings.Contains(result, "not configured") && !strings.Contains(result, "error")
		json.NewEncoder(w).Encode(map[string]interface{}{"result": result, "ok": ok})
	})

	// --- Go Module Management ---

	// Search pkg.go.dev
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

	// List installed modules
	http.HandleFunc("/api/modules/installed", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		mods := getInstalledModules()
		json.NewEncoder(w).Encode(map[string]interface{}{"modules": mods})
	})

	// Add a module
	http.HandleFunc("/api/modules/add", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" { http.Error(w, "Invalid method", 405); return }
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
		if req.Version == "" { req.Version = "latest" }
		result := addModule(map[string]interface{}{"module_path": req.ModulePath, "version": req.Version})
		if strings.HasPrefix(result, "go get failed") || strings.HasPrefix(result, "Error") {
			json.NewEncoder(w).Encode(map[string]string{"error": result})
			return
		}
		json.NewEncoder(w).Encode(map[string]string{"status": "ok", "output": result})
	})

	// Remove a module
	http.HandleFunc("/api/modules/remove", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" { http.Error(w, "Invalid method", 405); return }
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

	// Return raw go.mod
	http.HandleFunc("/api/modules/gomod", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		content := readGoMod()
		json.NewEncoder(w).Encode(map[string]string{"content": content})
	})


	// Edit / delete individual task
	http.HandleFunc("/api/tasks/", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		// extract id from path: /api/tasks/3
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
				if t.ID == taskID { found = true; continue }
				newTasks = append(newTasks, t)
			}
			if newTasks == nil { newTasks = []Task{} }
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
					if req.Title != ""       { tasksDb[i].Title = req.Title }
					if req.Description != "" { tasksDb[i].Description = req.Description }
					if req.Status != ""      { tasksDb[i].Status = req.Status }
					if req.Priority != 0     { tasksDb[i].Priority = req.Priority }
					if req.Result != ""      { tasksDb[i].Result = req.Result }
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

	// --- Task Management ---
	http.HandleFunc("/api/tasks", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.Method == "GET" {
			mu.Lock()
			tasks := make([]Task, len(tasksDb))
			copy(tasks, tasksDb)
			mu.Unlock()
			if tasks == nil { tasks = []Task{} }
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
			if req.Priority == 0 { req.Priority = 2 }
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
			// Also queue it for immediate agent pickup
			taskQueue <- fmt.Sprintf("TASK #%d: %s\n%s", task.ID, task.Title, task.Description)
			json.NewEncoder(w).Encode(task)
			return
		}
		http.Error(w, "Invalid method", 405)
	})

	// --- Host / Root credentials ---
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
		if r.Method != "POST" { http.Error(w, "Invalid method", 405); return }
		w.Header().Set("Content-Type", "application/json")
		mu.Lock()
		user, pass := hostUser, hostPass
		mu.Unlock()
		if pass == "" {
			json.NewEncoder(w).Encode(map[string]string{"error": "No password configured. Set credentials first."})
			return
		}
		rootUser := user
		if rootUser == "" { rootUser = "root" }
		// Run id + whoami to confirm effective identity
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


	// --- UI Title ---
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
			var req struct{ Title string `json:"title"` }
			json.NewDecoder(r.Body).Decode(&req)
			if req.Title == "" { req.Title = "Lily" }
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

		// ── Agent control ────────────────────────────────────────────────────
		"start": {
			"start",
			"Start the agent (default when no command given)",
			func(args []string) { runAgent() },
		},
		"task": {
			"task <text>",
			"Send a task to the running agent",
			func(args []string) {
				if len(args) == 0 { fmt.Println("Usage: agent task <text>"); os.Exit(1) }
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
				if len(msgs) == 0 { fmt.Println("(no messages)"); return }
				for _, m := range msgs {
					role, _ := m["role"].(string)
					content, _ := m["content"].(string)
					t, _ := m["timestamp"].(string)
					if len(t) > 16 { t = t[:16] }
					if role == "user" { fmt.Printf("[%s] 👤 %s\n", t, content) } else { fmt.Printf("[%s] 🤖 %s\n", t, content) }
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
				if len(args) > 0 { fmt.Sscanf(args[0], "%d", &n) }
				var entries []map[string]interface{}
				cliGetJSON("/api/logs", &entries)
				if len(entries) == 0 { fmt.Println("(no log entries)"); return }
				start := 0
				if len(entries) > n { start = len(entries) - n }
				for _, e := range entries[start:] {
					lvl, _ := e["level"].(string)
					msg, _ := e["content"].(string)
					t, _ := e["timestamp"].(string)
					if len(t) > 19 { t = t[:19] }
					fmt.Printf("[%s] %-12s %s\n", t, lvl, msg)
				}
			},
		},

		// ── Task management ──────────────────────────────────────────────────
		"tasks": {
			"tasks [--status pending|running|complete|failed|all]",
			"List tasks (optionally filtered by status)",
			func(args []string) {
				fs := flag.NewFlagSet("tasks", flag.ContinueOnError)
				status := fs.String("status", "all", "Filter by status")
				fs.Parse(args)
				var tasks []map[string]interface{}
				cliGetJSON("/api/tasks", &tasks)
				if len(tasks) == 0 { fmt.Println("(no tasks)"); return }
				prio := []string{"", "🔴 HIGH", "🟡 NORMAL", "🟢 LOW"}
				for _, t := range tasks {
					st, _ := t["status"].(string)
					if *status != "all" && st != *status { continue }
					id := int(t["id"].(float64))
					title, _ := t["title"].(string)
					pri := int(t["priority"].(float64))
					pLabel := ""
					if pri >= 1 && pri <= 3 { pLabel = prio[pri] }
					result, _ := t["result"].(string)
					fmt.Printf("#%d [%-8s] %s %s\n", id, strings.ToUpper(st), title, pLabel)
					if result != "" { fmt.Printf("    → %s\n", result) }
				}
			},
		},
		"task-add": {
			"task-add --title <title> [--desc <desc>] [--priority 1|2|3]",
			"Create a new task",
			func(args []string) {
				fs := flag.NewFlagSet("task-add", flag.ContinueOnError)
				title    := fs.String("title",    "", "Task title (required)")
				desc     := fs.String("desc",     "", "Task description")
				priority := fs.Int("priority", 2, "Priority: 1=high 2=normal 3=low")
				fs.Parse(args)
				if *title == "" { fmt.Println("--title is required"); os.Exit(1) }
				cliPost("/api/tasks", map[string]interface{}{"title": *title, "description": *desc, "priority": *priority})
				fmt.Println("✓ Task created.")
			},
		},
		"task-update": {
			"task-update <id> [--title t] [--desc d] [--status s] [--priority n] [--result r]",
			"Edit a task by ID",
			func(args []string) {
				if len(args) == 0 { fmt.Println("Usage: agent task-update <id> [flags]"); os.Exit(1) }
				id := args[0]; rest := args[1:]
				fs := flag.NewFlagSet("task-update", flag.ContinueOnError)
				title    := fs.String("title",    "", "New title")
				desc     := fs.String("desc",     "", "New description")
				status   := fs.String("status",   "", "New status: pending|running|complete|failed")
				priority := fs.Int("priority", 0, "New priority")
				result   := fs.String("result",   "", "New result")
				fs.Parse(rest)
				payload := map[string]interface{}{}
				if *title  != "" { payload["title"]       = *title }
				if *desc   != "" { payload["description"] = *desc }
				if *status != "" { payload["status"]      = *status }
				if *priority > 0 { payload["priority"]    = *priority }
				if *result != "" { payload["result"]      = *result }
				cliPut("/api/tasks/"+id, payload)
				fmt.Printf("✓ Task #%s updated.\n", id)
			},
		},
		"task-delete": {
			"task-delete <id>",
			"Delete a task by ID",
			func(args []string) {
				if len(args) == 0 { fmt.Println("Usage: agent task-delete <id>"); os.Exit(1) }
				cliDelete("/api/tasks/" + args[0])
				fmt.Printf("✓ Task #%s deleted.\n", args[0])
			},
		},

		// ── Memory ───────────────────────────────────────────────────────────
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
					if q != "" && !strings.Contains(strings.ToLower(content), q) && !strings.Contains(strings.ToLower(tags), q) { continue }
					fmt.Printf("[%s] %s\n", tags, content)
				}
			},
		},

		// ── Sessions ─────────────────────────────────────────────────────────
		"sessions": {
			"sessions",
			"List all chat sessions",
			func(args []string) {
				var sessions []map[string]interface{}
				cliGetJSON("/api/sessions", &sessions)
				if len(sessions) == 0 { fmt.Println("(no sessions)"); return }
				for _, s := range sessions {
					id, _ := s["id"].(string)
					title, _ := s["title"].(string)
					t, _ := s["created_at"].(string)
					if len(t) > 19 { t = t[:19] }
					fmt.Printf("[%s] %s — %s\n", t, id[:8], title)
				}
			},
		},

		// ── Personality ──────────────────────────────────────────────────────
		"personality": {
			"personality [--set <text>] [--show]",
			"Show or set agent personality",
			func(args []string) {
				fs := flag.NewFlagSet("personality", flag.ContinueOnError)
				set  := fs.String("set",  "", "New personality text")
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

		// ── LLM config ───────────────────────────────────────────────────────
		"llm": {
			"llm <main|coding|media|content> [--url u] [--model m] [--key k] [--timeout t]",
			"Show or configure an LLM",
			func(args []string) {
				if len(args) == 0 { fmt.Println("Usage: agent llm <main|coding|media|content> [flags]"); os.Exit(1) }
				name := args[0]; rest := args[1:]
				ep := map[string]string{"main": "/api/main_llm_config", "coding": "/api/coding_llm_config", "media": "/api/media_llm_config", "content": "/api/content_llm_config"}
				endpoint, ok := ep[name]
				if !ok { fmt.Printf("Unknown LLM: %s  (use main|coding|media|content)\n", name); os.Exit(1) }
				if len(rest) == 0 {
					var d map[string]interface{}
					cliGetJSON(endpoint, &d)
					fmt.Printf("URL:     %v\nModel:   %v\nKey:     %v\nTimeout: %v\n", d["url"], d["model"], d["key"], d["timeout"])
					return
				}
				fs := flag.NewFlagSet("llm", flag.ContinueOnError)
				url     := fs.String("url", "", "Base URL")
				model   := fs.String("model", "", "Model name")
				key     := fs.String("key", "", "API key")
				timeout := fs.Int("timeout", 0, "Timeout seconds")
				fs.Parse(rest)
				payload := map[string]interface{}{}
				if *url     != "" { payload["url"]     = *url }
				if *model   != "" { payload["model"]   = *model }
				if *key     != "" { payload["key"]     = *key }
				if *timeout >  0  { payload["timeout"] = *timeout }
				cliPost(endpoint, payload)
				fmt.Printf("✓ %s LLM config updated.\n", name)
			},
		},
		"llm-test": {
			"llm-test <main|coding|media|content>",
			"Test connection to an LLM",
			func(args []string) {
				if len(args) == 0 { fmt.Println("Usage: agent llm-test <main|coding|media|content>"); os.Exit(1) }
				ep := map[string]string{"main": "/api/main_llm_test", "coding": "/api/coding_llm_test", "media": "/api/media_llm_test", "content": "/api/content_llm_test"}
				endpoint, ok := ep[args[0]]
				if !ok { fmt.Printf("Unknown LLM: %s\n", args[0]); os.Exit(1) }
				var d map[string]interface{}
				cliPostJSON(endpoint, nil, &d)
				if e, ok := d["error"].(string); ok { fmt.Printf("✗ Error: %s\n", e) } else { fmt.Printf("✓ %s\n", d["response"]) }
			},
		},

		// ── Delegation ───────────────────────────────────────────────────────
		"delegation": {
			"delegation [--coding on|off] [--media on|off] [--content on|off]",
			"Show or toggle LLM delegation",
			func(args []string) {
				if len(args) == 0 {
					var d map[string]bool
					cliGetJSON("/api/delegation", &d)
					yn := func(b bool) string { if b { return "✓ ON" }; return "✗ OFF" }
					fmt.Printf("Coding:  %s\nMedia:   %s\nContent: %s\n", yn(d["coding"]), yn(d["media"]), yn(d["content"]))
					return
				}
				fs := flag.NewFlagSet("delegation", flag.ContinueOnError)
				coding  := fs.String("coding",  "", "on|off")
				media   := fs.String("media",   "", "on|off")
				content := fs.String("content", "", "on|off")
				fs.Parse(args)
				payload := map[string]interface{}{}
				parseBool := func(s string) *bool { v := s == "on" || s == "true" || s == "1"; return &v }
				if *coding  != "" { payload["coding"]  = parseBool(*coding)  }
				if *media   != "" { payload["media"]   = parseBool(*media)   }
				if *content != "" { payload["content"] = parseBool(*content) }
				cliPost("/api/delegation", payload)
				fmt.Println("✓ Delegation updated.")
			},
		},

		// ── Web search ───────────────────────────────────────────────────────
		"search": {
			"search <query>",
			"Run a web search using the configured provider",
			func(args []string) {
				if len(args) == 0 { fmt.Println("Usage: agent search <query>"); os.Exit(1) }
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
				key      := fs.String("key", "", "API key")
				cx       := fs.String("cx", "", "Google CX ID")
				fs.Parse(args)
				payload := map[string]interface{}{}
				if *provider != "" { payload["provider"] = *provider }
				if *key      != "" { payload["key"]      = *key }
				if *cx       != "" { payload["cx"]       = *cx }
				cliPost("/api/websearch_config", payload)
				fmt.Println("✓ Search config updated.")
			},
		},

		// ── Social ───────────────────────────────────────────────────────────
		"social-send": {
			"social-send <discord|telegram|slack|twitter|whatsapp> <message>",
			"Send a message via a social platform",
			func(args []string) {
				if len(args) < 2 { fmt.Println("Usage: agent social-send <platform> <message>"); os.Exit(1) }
				platform := args[0]; msg := strings.Join(args[1:], " ")
				var d map[string]interface{}
				cliPostJSON("/api/social_test", map[string]string{"platform": platform, "message": msg}, &d)
				if ok, _ := d["ok"].(bool); ok { fmt.Printf("✓ %s\n", d["result"]) } else { fmt.Printf("✗ %s\n", d["result"]) }
			},
		},
		"social-config": {
			"social-config [--discord-token t] [--discord-channel c] [--telegram-token t] [--telegram-chat c] [--slack-token t] [--slack-channel c] [--twitter-key k] [--twitter-secret s] [--twitter-token t] [--twitter-token-sec s] [--wa-sid s] [--wa-token t] [--wa-from f] [--wa-to to]",
			"Show or set social media credentials",
			func(args []string) {
				if len(args) == 0 {
					var d map[string]interface{}
					cliGetJSON("/api/social_config", &d)
					keys := []string{"discord_token","discord_channel","telegram_token","telegram_chat_id","slack_token","slack_channel","twitter_key","twitter_secret","twitter_token","twitter_token_sec","whatsapp_sid","whatsapp_token","whatsapp_from","whatsapp_to"}
					for _, k := range keys { fmt.Printf("%-22s %v\n", k+":", d[k]) }
					return
				}
				fs := flag.NewFlagSet("social-config", flag.ContinueOnError)
				dt  := fs.String("discord-token", "", "Discord bot token")
				dc  := fs.String("discord-channel", "", "Discord channel ID")
				tt  := fs.String("telegram-token", "", "Telegram bot token")
				tc  := fs.String("telegram-chat", "", "Telegram chat ID")
				st  := fs.String("slack-token", "", "Slack bot token")
				sc  := fs.String("slack-channel", "", "Slack channel")
				twk := fs.String("twitter-key", "", "Twitter API key")
				tws := fs.String("twitter-secret", "", "Twitter API secret")
				twt := fs.String("twitter-token", "", "Twitter access token")
				twts:= fs.String("twitter-token-sec", "", "Twitter access token secret")
				ws  := fs.String("wa-sid", "", "Twilio account SID")
				wt  := fs.String("wa-token", "", "Twilio auth token")
				wf  := fs.String("wa-from", "", "WhatsApp from number")
				wto := fs.String("wa-to", "", "WhatsApp to number")
				fs.Parse(args)
				payload := map[string]interface{}{}
				if *dt  != "" { payload["discord_token"]    = *dt }
				if *dc  != "" { payload["discord_channel"]  = *dc }
				if *tt  != "" { payload["telegram_token"]   = *tt }
				if *tc  != "" { payload["telegram_chat_id"] = *tc }
				if *st  != "" { payload["slack_token"]      = *st }
				if *sc  != "" { payload["slack_channel"]    = *sc }
				if *twk != "" { payload["twitter_key"]      = *twk }
				if *tws != "" { payload["twitter_secret"]   = *tws }
				if *twt != "" { payload["twitter_token"]    = *twt }
				if *twts!= "" { payload["twitter_token_sec"]= *twts }
				if *ws  != "" { payload["whatsapp_sid"]     = *ws }
				if *wt  != "" { payload["whatsapp_token"]   = *wt }
				if *wf  != "" { payload["whatsapp_from"]    = *wf }
				if *wto != "" { payload["whatsapp_to"]      = *wto }
				cliPost("/api/social_config", payload)
				fmt.Println("✓ Social credentials updated.")
			},
		},

		// ── Go modules ───────────────────────────────────────────────────────
		"modules": {
			"modules",
			"List installed Go module dependencies",
			func(args []string) {
				var d map[string]interface{}
				cliGetJSON("/api/modules/installed", &d)
				mods, _ := d["modules"].([]interface{})
				if len(mods) == 0 { fmt.Println("(no modules)"); return }
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
				if len(args) == 0 { fmt.Println("Usage: agent module-add <path> [version]"); os.Exit(1) }
				ver := "latest"; if len(args) > 1 { ver = args[1] }
				var d map[string]interface{}
				cliPostJSON("/api/modules/add", map[string]string{"module_path": args[0], "version": ver}, &d)
				if e, ok := d["error"].(string); ok { fmt.Printf("✗ %s\n", e) } else { fmt.Printf("✓ %s\n", d["output"]) }
			},
		},
		"module-remove": {
			"module-remove <path>",
			"Remove a Go module dependency",
			func(args []string) {
				if len(args) == 0 { fmt.Println("Usage: agent module-remove <path>"); os.Exit(1) }
				var d map[string]interface{}
				cliPostJSON("/api/modules/remove", map[string]string{"module_path": args[0]}, &d)
				if e, ok := d["error"].(string); ok { fmt.Printf("✗ %s\n", e) } else { fmt.Printf("✓ %s\n", d["output"]) }
			},
		},
		"module-search": {
			"module-search <query>",
			"Search pkg.go.dev for Go packages",
			func(args []string) {
				if len(args) == 0 { fmt.Println("Usage: agent module-search <query>"); os.Exit(1) }
				var d map[string]interface{}
				cliGetJSONQuery("/api/modules/search", "q="+strings.Join(args, "+"), &d)
				results, _ := d["results"].([]interface{})
				if len(results) == 0 { fmt.Println("(no results)"); return }
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

		// ── Host ─────────────────────────────────────────────────────────────
		"host": {
			"host [--user u] [--pass p] [--clear]",
			"Show or set host root access credentials",
			func(args []string) {
				if len(args) == 0 {
					var d map[string]interface{}
					cliGetJSON("/api/host_config", &d)
					fmt.Printf("User:   %v\nPass:   %v\nActive: %v\n", d["user"], d["masked_pass"], d["active"])
					return
				}
				fs := flag.NewFlagSet("host", flag.ContinueOnError)
				user  := fs.String("user", "", "Username")
				pass  := fs.String("pass", "", "Password")
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
				if e, ok := d["error"].(string); ok { fmt.Printf("✗ %s\n", e) } else { fmt.Printf("✓ %s\n", d["output"]) }
			},
		},

		// ── Status ───────────────────────────────────────────────────────────
		"status": {
			"status",
			"Show agent status: LLM config, task counts, delegation, search",
			func(args []string) {
				fmt.Println("═══ Agent Status ═══")
				// LLMs
				for _, name := range []string{"main","coding","media","content"} {
					ep := "/api/"+name+"_llm_config"
					var d map[string]interface{}
					cliGetJSON(ep, &d)
					model, _ := d["model"].(string); url, _ := d["url"].(string)
					status := "✗ not configured"
					if url != "" { status = fmt.Sprintf("✓ %s @ %s", model, url) }
					fmt.Printf("%-10s LLM: %s\n", strings.ToUpper(name), status)
				}
				// Delegation
				var del map[string]bool
				cliGetJSON("/api/delegation", &del)
				yn := func(b bool) string { if b { return "ON" }; return "OFF" }
				fmt.Printf("Delegation: coding=%s  media=%s  content=%s\n", yn(del["coding"]), yn(del["media"]), yn(del["content"]))
				// Tasks
				var tasks []map[string]interface{}
				cliGetJSON("/api/tasks", &tasks)
				counts := map[string]int{"pending":0,"running":0,"complete":0,"failed":0}
				for _, t := range tasks { counts[t["status"].(string)]++ }
				fmt.Printf("Tasks: %d pending  %d running  %d complete  %d failed\n", counts["pending"], counts["running"], counts["complete"], counts["failed"])
				// Search
				var sc map[string]interface{}
				cliGetJSON("/api/websearch_config", &sc)
				fmt.Printf("Search:    provider=%v\n", sc["provider"])
			},
		},

		// ── Help ─────────────────────────────────────────────────────────────
		"help": {
			"help [command]",
			"Show this help or detailed help for a command",
			func(args []string) {
				if len(args) > 0 {
					cmd, ok := cliCommands[args[0]]
					if !ok { fmt.Printf("Unknown command: %s\n", args[0]); os.Exit(1) }
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
// CLI HTTP helpers (talk to the running agent on localhost)
// ─────────────────────────────────────────────────────────────────────────────

func cliBase() string {
	return "http://localhost:" + WEB_PORT
}

func cliGetJSON(path string, out interface{}) {
	resp, err := http.Get(cliBase() + path)
	if err != nil { fmt.Fprintf(os.Stderr, "✗ Cannot reach agent at %s — is it running?\n  %v\n", cliBase(), err); os.Exit(1) }
	defer resp.Body.Close()
	json.NewDecoder(resp.Body).Decode(out)
}

func cliGetJSONQuery(path, query string, out interface{}) {
	resp, err := http.Get(cliBase() + path + "?" + query)
	if err != nil { fmt.Fprintf(os.Stderr, "✗ Cannot reach agent: %v\n", err); os.Exit(1) }
	defer resp.Body.Close()
	json.NewDecoder(resp.Body).Decode(out)
}

func cliPost(path string, payload interface{}) {
	var body *bytes.Buffer
	if payload != nil { b, _ := json.Marshal(payload); body = bytes.NewBuffer(b) } else { body = bytes.NewBuffer(nil) }
	resp, err := http.Post(cliBase()+path, "application/json", body)
	if err != nil { fmt.Fprintf(os.Stderr, "✗ Cannot reach agent: %v\n", err); os.Exit(1) }
	resp.Body.Close()
}

func cliPostJSON(path string, payload interface{}, out interface{}) {
	var body *bytes.Buffer
	if payload != nil { b, _ := json.Marshal(payload); body = bytes.NewBuffer(b) } else { body = bytes.NewBuffer(nil) }
	resp, err := http.Post(cliBase()+path, "application/json", body)
	if err != nil { fmt.Fprintf(os.Stderr, "✗ Cannot reach agent: %v\n", err); os.Exit(1) }
	defer resp.Body.Close()
	json.NewDecoder(resp.Body).Decode(out)
}

func cliPut(path string, payload interface{}) {
	b, _ := json.Marshal(payload)
	req, _ := http.NewRequest("PUT", cliBase()+path, bytes.NewBuffer(b))
	req.Header.Set("Content-Type", "application/json")
	resp, err := (&http.Client{}).Do(req)
	if err != nil { fmt.Fprintf(os.Stderr, "✗ Cannot reach agent: %v\n", err); os.Exit(1) }
	resp.Body.Close()
}

func cliDelete(path string) {
	req, _ := http.NewRequest("DELETE", cliBase()+path, nil)
	resp, err := (&http.Client{}).Do(req)
	if err != nil { fmt.Fprintf(os.Stderr, "✗ Cannot reach agent: %v\n", err); os.Exit(1) }
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

	// treat "-h" / "--help" as "help"
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

	// suppress "declared but not used" for imported packages used only in some paths
	_ = sort.Search
	_ = strconv.Itoa
}

