package chunk

import (
	"strings"
	"testing"
)

func TestTreeSitter_Python(t *testing.T) {
	// Check language registration
	lang := GetLanguageByExt(".py")
	if lang != nil {
		t.Logf("Python language registered: %s with %d node types", lang.Name, len(lang.NodeTypes))
	} else {
		t.Logf("Python language NOT registered")
	}

	content := `
def hello_world():
    """Say hello to the world."""
    print("Hello, world!")
    for i in range(10):
        print(f"Iteration {i}")
    return "done"

class Greeter:
    """A class that greets people."""

    def __init__(self, name):
        self.name = name

    def greet(self):
        return f"Hello, {self.name}"

def add(a, b):
    """Add two numbers together."""
    return a + b
`

	chunks, err := ChunkFile("/test/main.py", content, nil)
	if err != nil {
		t.Fatalf("ChunkFile failed: %v", err)
	}

	if len(chunks) == 0 {
		t.Fatal("expected chunks from Python file")
	}

	t.Logf("Got %d chunks:", len(chunks))
	for i, c := range chunks {
		t.Logf("  Chunk %d: %s (lines %d-%d)", i, c.Description, c.StartLine, c.EndLine)
	}

	var hasFunction, hasClass bool
	for _, c := range chunks {
		if strings.Contains(c.Description, "function") {
			hasFunction = true
		}
		if strings.Contains(c.Description, "class") {
			hasClass = true
		}
	}

	if !hasFunction {
		t.Error("expected to find Python function in chunks")
	}
	if !hasClass {
		t.Error("expected to find Python class in chunks")
	}
}

func TestTreeSitter_JavaScript(t *testing.T) {
	// Check language registration
	lang := GetLanguageByExt(".js")
	if lang == nil {
		t.Fatal("JavaScript language not registered")
	}
	t.Logf("Language registered: %s with %d node types", lang.Name, len(lang.NodeTypes))
	for _, nt := range lang.NodeTypes {
		t.Logf("  Node type: %s (%s)", nt.Type, nt.Kind)
	}

	content := `
function greetUser(name, greeting) {
    // This function greets a user with a custom greeting message
    const message = greeting + ", " + name + "!";
    console.log("Preparing to greet user...");
    console.log("Generated message: " + message);
    console.log("Greeting complete.");
    return message;
}

class UserService {
    constructor() {
        this.users = [];
        this.initialized = true;
        console.log("UserService initialized");
    }

    addUser(user) {
        console.log("Adding user: " + user.name);
        this.users.push(user);
        console.log("User added successfully");
        return this.users.length;
    }

    getUser(id) {
        console.log("Looking for user with id: " + id);
        return this.users.find(u => u.id === id);
    }
}
`

	chunks, err := ChunkFile("/test/app.js", content, nil)
	if err != nil {
		t.Fatalf("ChunkFile failed: %v", err)
	}

	if len(chunks) == 0 {
		t.Fatal("expected chunks from JavaScript file")
	}

	t.Logf("Got %d chunks:", len(chunks))
	for i, c := range chunks {
		t.Logf("  Chunk %d: %s (lines %d-%d)", i, c.Description, c.StartLine, c.EndLine)
	}

	var hasFunction, hasClass bool
	for _, c := range chunks {
		desc := strings.ToLower(c.Description)
		if strings.Contains(desc, "function") || strings.Contains(desc, "arrow") {
			hasFunction = true
		}
		if strings.Contains(desc, "class") {
			hasClass = true
		}
	}

	if !hasFunction {
		t.Error("expected to find JavaScript function in chunks")
	}
	if !hasClass {
		t.Error("expected to find JavaScript class in chunks")
	}
}

func TestTreeSitter_TypeScript(t *testing.T) {
	content := `
interface User {
    id: number;
    name: string;
    email: string;
}

type Status = 'active' | 'inactive' | 'pending';

function createUser(name: string, email: string): User {
    return {
        id: Math.random(),
        name,
        email
    };
}

class UserRepository {
    private users: User[] = [];

    add(user: User): void {
        this.users.push(user);
    }

    findById(id: number): User | undefined {
        return this.users.find(u => u.id === id);
    }
}
`

	chunks, err := ChunkFile("/test/user.ts", content, nil)
	if err != nil {
		t.Fatalf("ChunkFile failed: %v", err)
	}

	if len(chunks) == 0 {
		t.Fatal("expected chunks from TypeScript file")
	}

	t.Logf("Got %d chunks:", len(chunks))
	for i, c := range chunks {
		t.Logf("  Chunk %d: %s (lines %d-%d)", i, c.Description, c.StartLine, c.EndLine)
	}

	var hasFunction, hasClass, hasInterface bool
	for _, c := range chunks {
		if strings.Contains(c.Description, "function") {
			hasFunction = true
		}
		if strings.Contains(c.Description, "class") {
			hasClass = true
		}
		if strings.Contains(c.Description, "interface") {
			hasInterface = true
		}
	}

	if !hasFunction {
		t.Error("expected to find TypeScript function in chunks")
	}
	if !hasClass {
		t.Error("expected to find TypeScript class in chunks")
	}
	if !hasInterface {
		t.Error("expected to find TypeScript interface in chunks")
	}
}

func TestTreeSitter_Rust(t *testing.T) {
	content := `
fn main() {
    println!("Hello, world!");
    let result = add(5, 3);
    println!("Result: {}", result);
}

fn add(a: i32, b: i32) -> i32 {
    a + b
}

struct User {
    name: String,
    email: String,
    age: u32,
}

impl User {
    fn new(name: String, email: String, age: u32) -> Self {
        User { name, email, age }
    }

    fn greet(&self) -> String {
        format!("Hello, {}!", self.name)
    }
}

trait Greeter {
    fn greet(&self) -> String;
}
`

	chunks, err := ChunkFile("/test/main.rs", content, nil)
	if err != nil {
		t.Fatalf("ChunkFile failed: %v", err)
	}

	if len(chunks) == 0 {
		t.Fatal("expected chunks from Rust file")
	}

	var hasFunction, hasStruct, hasImpl bool
	for _, c := range chunks {
		if strings.Contains(c.Description, "function") {
			hasFunction = true
		}
		if strings.Contains(c.Description, "struct") {
			hasStruct = true
		}
		if strings.Contains(c.Description, "impl") {
			hasImpl = true
		}
	}

	if !hasFunction {
		t.Error("expected to find Rust function in chunks")
	}
	if !hasStruct {
		t.Error("expected to find Rust struct in chunks")
	}
	if !hasImpl {
		t.Error("expected to find Rust impl in chunks")
	}
}

func TestTreeSitter_Java(t *testing.T) {
	content := `
package com.example;

public class UserService {
    private List<User> users = new ArrayList<>();

    public UserService() {
        // Default constructor
    }

    public void addUser(User user) {
        users.add(user);
        System.out.println("User added: " + user.getName());
    }

    public User findUser(int id) {
        return users.stream()
            .filter(u -> u.getId() == id)
            .findFirst()
            .orElse(null);
    }
}

interface Repository<T> {
    void save(T entity);
    T findById(int id);
    List<T> findAll();
}
`

	chunks, err := ChunkFile("/test/UserService.java", content, nil)
	if err != nil {
		t.Fatalf("ChunkFile failed: %v", err)
	}

	if len(chunks) == 0 {
		t.Fatal("expected chunks from Java file")
	}

	var hasMethod, hasClass, hasInterface bool
	for _, c := range chunks {
		if strings.Contains(c.Description, "method") || strings.Contains(c.Description, "constructor") {
			hasMethod = true
		}
		if strings.Contains(c.Description, "class") {
			hasClass = true
		}
		if strings.Contains(c.Description, "interface") {
			hasInterface = true
		}
	}

	if !hasMethod {
		t.Error("expected to find Java method in chunks")
	}
	if !hasClass {
		t.Error("expected to find Java class in chunks")
	}
	if !hasInterface {
		t.Error("expected to find Java interface in chunks")
	}
}

func TestTreeSitter_C(t *testing.T) {
	content := `
#include <stdio.h>
#include <stdlib.h>

struct Point {
    int x;
    int y;
};

void print_point(struct Point* p) {
    printf("Point: (%d, %d)\n", p->x, p->y);
}

int add(int a, int b) {
    return a + b;
}

int main() {
    struct Point p = {10, 20};
    print_point(&p);
    printf("Sum: %d\n", add(5, 3));
    return 0;
}
`

	chunks, err := ChunkFile("/test/main.c", content, nil)
	if err != nil {
		t.Fatalf("ChunkFile failed: %v", err)
	}

	if len(chunks) == 0 {
		t.Fatal("expected chunks from C file")
	}

	var hasFunction, hasStruct bool
	for _, c := range chunks {
		if strings.Contains(c.Description, "function") {
			hasFunction = true
		}
		if strings.Contains(c.Description, "struct") {
			hasStruct = true
		}
	}

	if !hasFunction {
		t.Error("expected to find C function in chunks")
	}
	if !hasStruct {
		t.Error("expected to find C struct in chunks")
	}
}

func TestTreeSitter_CPP(t *testing.T) {
	content := `
#include <iostream>
#include <vector>
#include <string>

class User {
private:
    std::string name;
    int age;

public:
    User(const std::string& name, int age)
        : name(name), age(age) {}

    std::string getName() const {
        return name;
    }

    void setName(const std::string& newName) {
        name = newName;
    }
};

namespace utils {
    int add(int a, int b) {
        return a + b;
    }

    void print(const std::string& msg) {
        std::cout << msg << std::endl;
    }
}

int main() {
    User user("Alice", 30);
    std::cout << "Hello, " << user.getName() << std::endl;
    return 0;
}
`

	chunks, err := ChunkFile("/test/main.cpp", content, nil)
	if err != nil {
		t.Fatalf("ChunkFile failed: %v", err)
	}

	if len(chunks) == 0 {
		t.Fatal("expected chunks from C++ file")
	}

	var hasFunction, hasClass, hasNamespace bool
	for _, c := range chunks {
		if strings.Contains(c.Description, "function") {
			hasFunction = true
		}
		if strings.Contains(c.Description, "class") {
			hasClass = true
		}
		if strings.Contains(c.Description, "namespace") {
			hasNamespace = true
		}
	}

	if !hasFunction {
		t.Error("expected to find C++ function in chunks")
	}
	if !hasClass {
		t.Error("expected to find C++ class in chunks")
	}
	if !hasNamespace {
		t.Error("expected to find C++ namespace in chunks")
	}
}

func TestTreeSitter_QualifiedSymbols(t *testing.T) {
	tests := []struct {
		name    string
		path    string
		content string
		want    string
	}{
		{
			name: "python class method",
			path: "/test/worker.py",
			content: `class Worker:
    def execute(self, task):
        normalized = task.strip().lower()
        result = self.process(normalized)
        return result or "empty"
`,
			want: "Python function Worker.execute",
		},
		{
			name: "javascript class method",
			path: "/test/worker.js",
			content: `class Worker {
    execute(task) {
        const normalized = task.trim().toLowerCase();
        const result = this.process(normalized);
        return result || "empty";
    }
}
`,
			want: "Javascript method Worker.execute",
		},
		{
			name: "typescript namespace method",
			path: "/test/worker.ts",
			content: `namespace Platform {
    export class Worker {
        execute(task: string): string {
            const normalized = task.trim().toLowerCase();
            const result = this.process(normalized);
            return result || "empty";
        }
    }
}
`,
			want: "Typescript method Platform.Worker.execute",
		},
		{
			name: "rust module impl method",
			path: "/test/worker.rs",
			content: `mod jobs {
    struct Worker;

    impl Worker {
        fn execute(&self, task: &str) -> String {
            let normalized = task.trim().to_lowercase();
            format!("processed: {}", normalized)
        }
    }
}
`,
			want: "Rust function jobs.Worker.execute",
		},
		{
			name: "java class method",
			path: "/test/Worker.java",
			content: `class Worker {
    String execute(String task) {
        String normalized = task.trim().toLowerCase();
        String result = process(normalized);
        return result.isEmpty() ? "empty" : result;
    }
}
`,
			want: "Java method Worker.execute",
		},
		{
			name: "C++ namespace class method",
			path: "/test/worker.cpp",
			content: `namespace jobs {
class Worker {
public:
    std::string execute(const std::string& task) {
        const auto normalized = normalize(task);
        return process(normalized);
    }
};
}
`,
			want: "Cpp function jobs.Worker.execute",
		},
		{
			name: "C function declarator",
			path: "/test/worker.c",
			content: `const char *process_request(const char *request) {
    const char *normalized = normalize_request(request);
    log_request(normalized);
    return normalized;
}
`,
			want: "C function process_request",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			chunks, err := ChunkFile(tt.path, tt.content, nil)
			if err != nil {
				t.Fatalf("ChunkFile failed: %v", err)
			}
			if findChunk(chunks, tt.want) == nil {
				t.Fatalf("missing description %q in chunks: %+v", tt.want, chunks)
			}
		})
	}
}

func TestTreeSitter_AssignedJavaScriptFunctions(t *testing.T) {
	content := `/** Handles an incoming request. */
const handleRequest = async (request) => {
    const normalized = await normalizeRequest(request);
    return dispatchRequest(normalized);
};

const handlers = {
    onError: (error) => {
        console.error("request failed", error);
        return recoverFromError(error);
    }
};

class Controller {
    handle = (request) => {
        const normalized = normalizeRequest(request);
        return this.dispatch(normalized);
    };
}
`

	chunks, err := ChunkFile("/test/handlers.ts", content, nil)
	if err != nil {
		t.Fatalf("ChunkFile failed: %v", err)
	}

	tests := []struct {
		description string
		content     string
	}{
		{"Typescript arrow function handleRequest", "const handleRequest"},
		{"Typescript arrow function onError", "onError:"},
		{"Typescript arrow function Controller.handle", "handle ="},
	}
	for _, tt := range tests {
		chunk := findChunk(chunks, tt.description)
		if chunk == nil {
			t.Errorf("missing description %q in chunks: %+v", tt.description, chunks)
			continue
		}
		if !strings.Contains(chunk.Content, tt.content) {
			t.Errorf("chunk %q does not include assigned name %q", chunk.Content, tt.content)
		}
	}

	handler := findChunk(chunks, "Typescript arrow function handleRequest")
	if handler == nil {
		t.Fatal("missing handleRequest chunk")
	}
	if handler.StartLine != 1 || !strings.HasPrefix(handler.Content, "/** Handles") {
		t.Errorf("JSDoc was not included in assigned function chunk: %+v", handler)
	}
	if !strings.Contains(handler.Description, "Handles an incoming request") {
		t.Errorf("JSDoc missing from description: %q", handler.Description)
	}
}

func TestTreeSitter_TSXUsesTSXGrammar(t *testing.T) {
	content := `interface CardProps {
    title: string;
    details: string;
}

export function Card(props: CardProps) {
    const heading = props.title.toUpperCase();
    return <article><h2>{heading}</h2><p>{props.details}</p></article>;
}
`

	chunks, err := ChunkFile("/test/card.tsx", content, nil)
	if err != nil {
		t.Fatalf("ChunkFile failed: %v", err)
	}
	chunk := findChunk(chunks, "Typescript function Card")
	if chunk == nil {
		t.Fatalf("TSX function was not parsed semantically: %+v", chunks)
	}
	if !strings.Contains(chunk.Content, "<article>") {
		t.Errorf("TSX content missing from function chunk: %q", chunk.Content)
	}
}

func TestTreeSitter_DocumentationComments(t *testing.T) {
	tests := []struct {
		name       string
		path       string
		content    string
		wantChunk  string
		wantDoc    string
		wantPrefix string
	}{
		{
			name:       "JavaScript JSDoc",
			path:       "/test/load.js",
			content:    "/** Loads account data from durable storage. */\nfunction loadAccount(id) {\n    const account = database.find(id);\n    auditAccess(account);\n    return account;\n}\n",
			wantChunk:  "Javascript function loadAccount",
			wantDoc:    "Loads account data from durable storage",
			wantPrefix: "/**",
		},
		{
			name:       "Rust line docs",
			path:       "/test/load.rs",
			content:    "/// Loads account data\n/// from durable storage.\nfn load_account(id: u64) -> Account {\n    let account = database_find(id);\n    audit_access(&account);\n    account\n}\n",
			wantChunk:  "Rust function load_account",
			wantDoc:    "Loads account data from durable storage",
			wantPrefix: "///",
		},
		{
			name:       "Java Javadoc",
			path:       "/test/Accounts.java",
			content:    "/** Loads account data from durable storage. */\nclass Accounts {\n    Account load(long id) {\n        Account account = database.find(id);\n        audit.access(account);\n        return account;\n    }\n}\n",
			wantChunk:  "Java class Accounts",
			wantDoc:    "Loads account data from durable storage",
			wantPrefix: "/**",
		},
		{
			name:       "C Doxygen",
			path:       "/test/load.c",
			content:    "/** Loads account data from durable storage. */\nstruct account *load_account(long id) {\n    struct account *value = database_find(id);\n    audit_access(value);\n    return value;\n}\n",
			wantChunk:  "C function load_account",
			wantDoc:    "Loads account data from durable storage",
			wantPrefix: "/**",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			chunks, err := ChunkFile(tt.path, tt.content, nil)
			if err != nil {
				t.Fatalf("ChunkFile failed: %v", err)
			}
			chunk := findChunk(chunks, tt.wantChunk)
			if chunk == nil {
				t.Fatalf("missing description %q in chunks: %+v", tt.wantChunk, chunks)
			}
			if chunk.StartLine != 1 || !strings.HasPrefix(chunk.Content, tt.wantPrefix) {
				t.Errorf("documentation comment not included in chunk: %+v", chunk)
			}
			if !strings.Contains(chunk.Description, tt.wantDoc) {
				t.Errorf("description %q does not include %q", chunk.Description, tt.wantDoc)
			}
		})
	}
}

func TestTreeSitter_DocumentationCommentsThroughWrappers(t *testing.T) {
	content := `/** Loads account data. */
/** Includes durable history. */
export function loadAccount(accountId: string, includeHistory: boolean) {
    const account = database.load(accountId);
    return includeHistory ? attachHistory(account) : account;
}

/** Handles an incoming request. */
export const handleRequest = async (request: Request) => {
    const normalized = await normalizeRequest(request);
    return dispatchRequest(normalized);
};

/** Resolves an ambient account declaration. */
declare function resolveAccount(accountId: string, tenantId: string, includeHistory: boolean, includeMetadata: boolean): Promise<Account>;

/** Resolves an exported ambient account declaration. */
export declare function resolveExportedAccount(accountId: string, tenantId: string, includeHistory: boolean): Promise<Account>;
`

	chunks, err := ChunkFile("/test/accounts.ts", content, nil)
	if err != nil {
		t.Fatalf("ChunkFile failed: %v", err)
	}

	load := findChunk(chunks, "Typescript function loadAccount")
	if load == nil {
		t.Fatalf("missing exported function chunk: %+v", chunks)
	}
	if !strings.HasPrefix(load.Content, "/** Loads account data. */") ||
		!strings.Contains(load.Content, "export function loadAccount") {
		t.Errorf("exported function chunk does not include its docs and wrapper: %q", load.Content)
	}
	if !strings.Contains(load.Description, "Loads account data. Includes durable history.") {
		t.Errorf("consecutive documentation comments were not cleaned: %q", load.Description)
	}
	if strings.Contains(load.Description, "*/") || strings.Contains(load.Description, "/**") {
		t.Errorf("documentation markers leaked into description: %q", load.Description)
	}

	handler := findChunk(chunks, "Typescript arrow function handleRequest")
	if handler == nil {
		t.Fatalf("missing exported arrow function chunk: %+v", chunks)
	}
	if !strings.HasPrefix(handler.Content, "/** Handles an incoming request. */") ||
		!strings.Contains(handler.Content, "export const handleRequest") {
		t.Errorf("exported arrow chunk does not include its docs and wrapper: %q", handler.Content)
	}
	if !strings.Contains(handler.Description, "Handles an incoming request.") {
		t.Errorf("exported arrow description does not include docs: %q", handler.Description)
	}

	ambient := findChunk(chunks, "Typescript function resolveAccount")
	if ambient == nil {
		t.Fatalf("missing ambient function chunk: %+v", chunks)
	}
	if !strings.HasPrefix(ambient.Content, "/** Resolves an ambient account declaration. */") ||
		!strings.Contains(ambient.Content, "declare function resolveAccount") {
		t.Errorf("ambient function chunk does not include its docs and wrapper: %q", ambient.Content)
	}

	exportedAmbient := findChunk(chunks, "Typescript function resolveExportedAccount")
	if exportedAmbient == nil {
		t.Fatalf("missing exported ambient function chunk: %+v", chunks)
	}
	if !strings.HasPrefix(exportedAmbient.Content, "/** Resolves an exported ambient account declaration. */") ||
		!strings.Contains(exportedAmbient.Content, "export declare function resolveExportedAccount") {
		t.Errorf("exported ambient function chunk does not include its docs and wrappers: %q", exportedAmbient.Content)
	}
}

func TestTreeSitter_MultipleAssignedFunctionsUseDistinctRanges(t *testing.T) {
	content := `const onSuccess = (response) => {
    const normalized = normalizeResponse(response);
    return publishResponse(normalized);
}, onError = (error) => {
    const recovered = recoverFromError(error);
    return publishFailure(recovered);
};
`

	chunks, err := ChunkFile("/test/callbacks.js", content, nil)
	if err != nil {
		t.Fatalf("ChunkFile failed: %v", err)
	}

	success := findChunk(chunks, "Javascript arrow function onSuccess")
	failure := findChunk(chunks, "Javascript arrow function onError")
	if success == nil || failure == nil {
		t.Fatalf("missing assigned function chunks: %+v", chunks)
	}
	if strings.Contains(success.Content, "onError") {
		t.Errorf("onSuccess chunk includes onError declarator: %q", success.Content)
	}
	if strings.Contains(failure.Content, "onSuccess") {
		t.Errorf("onError chunk includes onSuccess declarator: %q", failure.Content)
	}
	if success.StartLine == failure.StartLine && success.EndLine == failure.EndLine {
		t.Errorf("assigned functions share the same range: success=%+v failure=%+v", success, failure)
	}
}

func TestTreeSitter_WrappersProduceOneSemanticChunk(t *testing.T) {
	tests := []struct {
		name        string
		path        string
		content     string
		description string
		wrapper     string
	}{
		{
			name: "Python decorator",
			path: "/test/service.py",
			content: `class Service:
    @cached(ttl=30)
    def fetch(self, account_id):
        """Fetch an account from durable storage."""
        account = self.database.find(account_id)
        return account or self.empty_account()
`,
			description: "Python function Service.fetch",
			wrapper:     "@cached",
		},
		{
			name: "C++ template",
			path: "/test/choose.cpp",
			content: `template <typename T>
T choose_first(T left, T right) {
    const auto selected = left.valid() ? left : right;
    log_selection(selected);
    return selected;
}
`,
			description: "Cpp function choose_first",
			wrapper:     "template <typename T>",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			chunks, err := ChunkFile(tt.path, tt.content, nil)
			if err != nil {
				t.Fatalf("ChunkFile failed: %v", err)
			}
			matches := 0
			for i := range chunks {
				if strings.Contains(chunks[i].Description, tt.description) {
					matches++
					if !strings.Contains(chunks[i].Content, tt.wrapper) {
						t.Errorf("wrapper %q missing from chunk content %q", tt.wrapper, chunks[i].Content)
					}
				}
			}
			if matches != 1 {
				t.Errorf("got %d chunks for %q, want exactly one: %+v", matches, tt.description, chunks)
			}
		})
	}
}

func TestTreeSitter_ExpandedDeclarations(t *testing.T) {
	tests := []struct {
		name    string
		path    string
		content string
		want    string
	}{
		{"JavaScript generator", "/test/ids.js", "function* generateIds(start) {\n    let current = start;\n    while (current < start + 10) {\n        yield current++;\n    }\n}\n", "Javascript generator function generateIds"},
		{"TypeScript enum", "/test/status.ts", "enum RequestStatus {\n    Pending = 'pending',\n    Running = 'running',\n    Complete = 'complete',\n    Failed = 'failed',\n}\n", "Typescript enum RequestStatus"},
		{"Rust macro", "/test/macros.rs", "macro_rules! log_result {\n    ($value:expr) => {\n        println!(\"result: {:?}\", $value);\n        record_metric($value);\n    };\n}\n", "Rust macro log_result"},
		{"C union", "/test/value.c", "union Value {\n    long integer_value;\n    double decimal_value;\n    const char *string_value;\n    void *pointer_value;\n};\n", "C union Value"},
		{"C++ concept", "/test/addable.cpp", "template <typename T>\nconcept Addable = requires(T value) {\n    value + value;\n    value += value;\n    normalize(value);\n};\n", "Cpp concept Addable"},
		{"Java record", "/test/Account.java", "record Account(long id, String name, String email) {\n    String displayName() {\n        return name + \" <\" + email + \">\";\n    }\n}\n", "Java record Account"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			chunks, err := ChunkFile(tt.path, tt.content, nil)
			if err != nil {
				t.Fatalf("ChunkFile failed: %v", err)
			}
			if findChunk(chunks, tt.want) == nil {
				t.Fatalf("missing description %q in chunks: %+v", tt.want, chunks)
			}
		})
	}
}

func TestTreeSitter_SyntaxError(t *testing.T) {
	// Python with syntax error should fall back to size-based
	content := `
def broken(value:
    """This declaration is long enough to otherwise become a semantic chunk."""
    normalized = str(value).strip().lower()
    record_value(normalized)
    return normalized
`

	chunks, err := ChunkFile("/test/broken.py", content, nil)
	if err != nil {
		t.Fatalf("ChunkFile failed: %v", err)
	}

	if len(chunks) == 0 {
		t.Error("expected fallback to size-based chunking for syntax errors")
	}
	for _, chunk := range chunks {
		if !strings.HasPrefix(chunk.Description, "Code from broken.py") {
			t.Errorf("syntax error produced semantic chunk instead of fallback: %+v", chunk)
		}
	}
}

func findChunk(chunks []Chunk, description string) *Chunk {
	for i := range chunks {
		if strings.Contains(chunks[i].Description, description) {
			return &chunks[i]
		}
	}
	return nil
}

func TestLanguageRegistry(t *testing.T) {
	tests := []struct {
		ext      string
		expected string
	}{
		{".py", "python"},
		{".js", "javascript"},
		{".ts", "typescript"},
		{".rs", "rust"},
		{".java", "java"},
		{".c", "c"},
		{".cpp", "cpp"},
	}

	for _, tt := range tests {
		lang := GetLanguageByExt(tt.ext)
		if lang == nil {
			t.Errorf("expected language for %s", tt.ext)
			continue
		}
		if lang.Name != tt.expected {
			t.Errorf("expected %s for %s, got %s", tt.expected, tt.ext, lang.Name)
		}
	}
}

func TestLanguageRegistry_Unknown(t *testing.T) {
	lang := GetLanguageByExt(".unknown")
	if lang != nil {
		t.Error("expected nil for unknown extension")
	}
}

func BenchmarkTreeSitter_Python(b *testing.B) {
	content := `
def process_data(data):
    """Process incoming data."""
    result = []
    for item in data:
        if item.valid:
            result.append(transform(item))
    return result

class DataProcessor:
    def __init__(self):
        self.cache = {}

    def process(self, key, value):
        if key in self.cache:
            return self.cache[key]
        result = self._compute(value)
        self.cache[key] = result
        return result
`

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = ChunkFile("/test/processor.py", content, nil)
	}
}

func BenchmarkTreeSitter_TypeScript(b *testing.B) {
	content := `
interface Config {
    host: string;
    port: number;
}

class Server {
    constructor(private config: Config) {}

    start(): void {
        console.log('Starting server...');
    }
}

function createServer(config: Config): Server {
    return new Server(config);
}
`

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = ChunkFile("/test/server.ts", content, nil)
	}
}

type treeSitterQualityExpectation struct {
	description          string
	contentContains      string
	contentExcludes      string
	contentPrefix        string
	forbiddenDescription string
	exactMatches         int
	retained             bool
}

type treeSitterQualityFixture struct {
	path         string
	content      string
	expectations []treeSitterQualityExpectation
}

var benchmarkTreeSitterChunkCount int

// BenchmarkTreeSitter_MultilingualQuality measures semantic coverage, chunking
// cost, and downstream embedding volume for a fixed corpus spanning every
// tree-sitter language. retained_pct guards established behavior; enhanced_pct
// tracks newer semantic capabilities. This does not measure embedding latency.
func BenchmarkTreeSitter_MultilingualQuality(b *testing.B) {
	fixtures := multilingualQualityFixtures()

	b.StopTimer()
	retainedPassed, retainedTotal := 0, 0
	enhancedPassed, enhancedTotal := 0, 0
	chunkCount := 0
	embeddingBytes := 0
	embeddingTokens := 0
	for _, fixture := range fixtures {
		chunks, err := ChunkFile(fixture.path, fixture.content, nil)
		if err != nil {
			b.Fatalf("ChunkFile(%s): %v", fixture.path, err)
		}
		chunkCount += len(chunks)
		for _, chunk := range chunks {
			embeddingText := chunk.Content
			if chunk.Description != "" {
				embeddingText = chunk.Description + "\n\n" + chunk.Content
			}
			embeddingBytes += len(embeddingText)
			embeddingTokens += estimateTokens(embeddingText)
		}
		for _, expectation := range fixture.expectations {
			passed := treeSitterExpectationMet(chunks, expectation)
			if expectation.retained {
				retainedTotal++
				if passed {
					retainedPassed++
				}
			} else {
				enhancedTotal++
				if passed {
					enhancedPassed++
				}
			}
		}
	}
	b.ReportMetric(100*float64(retainedPassed)/float64(retainedTotal), "retained_pct")
	b.ReportMetric(100*float64(enhancedPassed)/float64(enhancedTotal), "enhanced_pct")
	b.ReportMetric(
		100*float64(retainedPassed+enhancedPassed)/float64(retainedTotal+enhancedTotal),
		"quality_pct",
	)
	b.ReportMetric(float64(len(fixtures)), "files/op")
	b.ReportMetric(float64(chunkCount), "chunks/op")
	b.ReportMetric(float64(chunkCount), "embed_inputs/op")
	b.ReportMetric(float64(embeddingBytes), "embed_bytes/op")
	b.ReportMetric(float64(embeddingTokens), "embed_tokens/op")

	b.StartTimer()
	for i := 0; i < b.N; i++ {
		chunkCount := 0
		for _, fixture := range fixtures {
			chunks, err := ChunkFile(fixture.path, fixture.content, nil)
			if err != nil {
				b.Fatalf("ChunkFile(%s): %v", fixture.path, err)
			}
			chunkCount += len(chunks)
		}
		benchmarkTreeSitterChunkCount = chunkCount
	}
}

func treeSitterExpectationMet(chunks []Chunk, expectation treeSitterQualityExpectation) bool {
	if expectation.forbiddenDescription != "" {
		for _, chunk := range chunks {
			if strings.Contains(chunk.Description, expectation.forbiddenDescription) {
				return false
			}
		}
		return true
	}

	matches := 0
	for _, chunk := range chunks {
		if !strings.Contains(chunk.Description, expectation.description) {
			continue
		}
		if expectation.contentContains != "" && !strings.Contains(chunk.Content, expectation.contentContains) {
			continue
		}
		if expectation.contentExcludes != "" && strings.Contains(chunk.Content, expectation.contentExcludes) {
			continue
		}
		if expectation.contentPrefix != "" && !strings.HasPrefix(chunk.Content, expectation.contentPrefix) {
			continue
		}
		matches++
	}
	if expectation.exactMatches > 0 {
		return matches == expectation.exactMatches
	}
	return matches > 0
}

func multilingualQualityFixtures() []treeSitterQualityFixture {
	return []treeSitterQualityFixture{
		{
			path: "/bench/service.py",
			content: `def normalize(value):
    cleaned = str(value).strip().lower()
    return cleaned or "empty"

class Service:
    @cached(ttl=30)
    def fetch(self, account_id):
        """Fetch an account from durable storage."""
        account = self.database.find(account_id)
        return account or self.empty_account()
`,
			expectations: []treeSitterQualityExpectation{
				{description: "Python function normalize", retained: true},
				{description: "Python class Service", retained: true},
				{description: "Python function Service.fetch"},
				{description: "Python function Service.fetch", contentContains: "@cached", exactMatches: 1},
				{forbiddenDescription: "Python decorated"},
			},
		},
		{
			path: "/bench/handlers.js",
			content: `function normalize(value) {
    const cleaned = String(value).trim().toLowerCase();
    return cleaned || "empty";
}

class Controller {
    execute(request) {
        const normalized = normalize(request);
        return this.dispatch(normalized);
    }

    handle = (request) => {
        const normalized = normalizeRequest(request);
        return this.dispatch(normalized);
    };
}

/** Handles an incoming request. */
/** Dispatches its normalized form. */
export const handleRequest = async (request) => {
    const normalized = await normalizeRequest(request);
    return dispatchRequest(normalized);
};

const handlers = {
    onError: (error) => {
        console.error("request failed", error);
        return recoverFromError(error);
    },
};

const recover = function(error) {
    console.warn("recovering request", error);
    return recoverFromError(error);
};

const onSuccess = (response) => {
    const normalized = normalizeResponse(response);
    return publishResponse(normalized);
}, onFailure = (error) => {
    const recovered = recoverFromError(error);
    return publishFailure(recovered);
};

function* generateIds(start) {
    let current = start;
    while (current < start + 10) yield current++;
}

const generateMoreIds = function* (start) {
    let current = start;
    while (current < start + 10) yield current++;
};
`,
			expectations: []treeSitterQualityExpectation{
				{description: "Javascript function normalize", retained: true},
				{description: "Javascript class Controller", retained: true},
				{description: "Javascript method Controller.execute"},
				{description: "Javascript arrow function handleRequest", contentContains: "const handleRequest"},
				{description: "Javascript arrow function handleRequest", contentPrefix: "/** Handles an incoming request. */"},
				{description: "Javascript arrow function handleRequest", contentContains: "dispatchRequest(normalized)"},
				{description: "Javascript arrow function handleRequest in handlers.js. Handles an incoming request. Dispatches its normalized form."},
				{description: "Javascript arrow function onError", contentContains: "onError:"},
				{description: "Javascript arrow function Controller.handle", contentContains: "handle ="},
				{description: "Javascript function recover", contentContains: "const recover"},
				{description: "Javascript arrow function onSuccess", contentExcludes: "onFailure"},
				{description: "Javascript arrow function onFailure", contentExcludes: "onSuccess"},
				{description: "Javascript generator function generateIds"},
				{description: "Javascript generator function generateMoreIds", contentContains: "const generateMoreIds"},
			},
		},
		{
			path: "/bench/platform.ts",
			content: `interface Request {
    id: number;
    payload: string;
}

abstract class BaseWorker {
    abstract execute(request: Request): string;
}

enum RequestStatus {
    Pending = "pending",
    Complete = "complete",
}

namespace Platform {
    export class Worker {
        execute(request: Request): string {
            const normalized = request.payload.trim();
            return normalized || String(request.id);
        }
    }
}

/** Resolves an ambient account declaration. */
declare function resolveAccount(accountId: string, tenantId: string, includeHistory: boolean, includeMetadata: boolean): Promise<Account>;

/** Resolves an exported ambient account declaration. */
export declare function resolveExportedAccount(accountId: string, tenantId: string, includeHistory: boolean): Promise<Account>;
`,
			expectations: []treeSitterQualityExpectation{
				{description: "Typescript interface Request", retained: true},
				{description: "Typescript class BaseWorker"},
				{description: "Typescript enum RequestStatus"},
				{description: "Typescript namespace Platform"},
				{description: "Typescript method Platform.Worker.execute"},
				{description: "Typescript function resolveAccount", contentContains: "declare function resolveAccount"},
				{description: "Typescript function resolveAccount in platform.ts. Resolves an ambient account declaration."},
				{description: "Typescript function resolveExportedAccount", contentContains: "export declare function resolveExportedAccount"},
				{description: "Typescript function resolveExportedAccount in platform.ts. Resolves an exported ambient account declaration."},
			},
		},
		{
			path: "/bench/card.tsx",
			content: `interface CardProps {
    title: string;
    details: string;
}

export function Card(props: CardProps) {
    const heading = props.title.toUpperCase();
    return <article><h2>{heading}</h2><p>{props.details}</p></article>;
}
`,
			expectations: []treeSitterQualityExpectation{
				{description: "Typescript function Card", contentContains: "<article>"},
			},
		},
		{
			path: "/bench/jobs.rs",
			content: `struct Account {
    id: u64,
    name: String,
}

mod jobs {
    struct Worker;

    impl Worker {
        /// Execute a queued job.
        fn execute(&self, task: &str) -> String {
            let normalized = task.trim().to_lowercase();
            format!("processed: {}", normalized)
        }
    }
}

union Number {
    integer: i64,
    decimal: f64,
}

type AccountId = u64;

macro_rules! record_result {
    ($value:expr) => { println!("result: {:?}", $value) };
}
`,
			expectations: []treeSitterQualityExpectation{
				{description: "Rust struct Account", retained: true},
				{description: "Rust function jobs.Worker.execute"},
				{description: "Rust function jobs.Worker.execute", contentPrefix: "/// Execute a queued job."},
				{description: "Rust function jobs.Worker.execute in jobs.rs. Execute a queued job"},
				{description: "Rust union Number"},
				{description: "Rust type AccountId"},
				{description: "Rust module jobs"},
				{description: "Rust macro record_result"},
			},
		},
		{
			path: "/bench/accounts.c",
			content: `struct account {
    long id;
    const char *name;
};

union value {
    long integer_value;
    double decimal_value;
    const char *string_value;
};

/** Load account data from durable storage. */
struct account *load_account(long id) {
    struct account *value = database_find(id);
    audit_access(value);
    return value;
}
`,
			expectations: []treeSitterQualityExpectation{
				{description: "C struct account", retained: true},
				{description: "C union value"},
				{description: "C function load_account"},
				{description: "C function load_account", contentPrefix: "/** Load account data"},
				{description: "C function load_account in accounts.c. Load account data from durable storage"},
			},
		},
		{
			path: "/bench/worker.cpp",
			content: `class Account {
public:
    long id() const { return account_id; }
private:
    long account_id;
};

namespace jobs {
class Worker {
public:
    std::string execute(const std::string& task) {
        const auto normalized = normalize(task);
        return process(normalized);
    }
};
}

template <typename T>
T choose_first(T left, T right) {
    const auto selected = left.valid() ? left : right;
    return selected;
}

template <typename T>
concept Addable = requires(T value) {
    value + value;
    value += value;
};

using AccountId = unsigned long;
`,
			expectations: []treeSitterQualityExpectation{
				{description: "Cpp class Account", retained: true},
				{description: "Cpp function jobs.Worker.execute"},
				{description: "Cpp function choose_first", contentContains: "template <typename T>", exactMatches: 1},
				{forbiddenDescription: "Cpp template"},
				{description: "Cpp concept Addable"},
				{description: "Cpp type AccountId"},
			},
		},
		{
			path: "/bench/Accounts.java",
			content: `class LegacyAccount {
    private long id;
    long id() { return id; }
}

/** Provides account operations. */
class Accounts {
    Account load(long id) {
        Account account = database.find(id);
        audit.access(account);
        return account;
    }
}

record Account(long id, String name, String email) {
    String displayName() { return name + " <" + email + ">"; }
}

@interface Audited {
    String value() default "account";
}
`,
			expectations: []treeSitterQualityExpectation{
				{description: "Java class LegacyAccount", retained: true},
				{description: "Java method Accounts.load"},
				{description: "Java class Accounts", contentPrefix: "/** Provides account operations. */"},
				{description: "Java class Accounts in Accounts.java. Provides account operations"},
				{description: "Java record Account"},
				{description: "Java annotation Audited"},
			},
		},
		{
			path: "/bench/broken.py",
			content: `def broken(value:
    """This invalid declaration must use size-based fallback."""
    normalized = str(value).strip().lower()
    record_value(normalized)
    return normalized
`,
			expectations: []treeSitterQualityExpectation{
				{description: "Code from broken.py"},
			},
		},
	}
}

func TestRegistry_Debug(t *testing.T) {
	t.Logf("Registry has %d languages", len(Registry))
	for name, cfg := range Registry {
		t.Logf("  - %s: extensions=%v, nodeTypes=%d", name, cfg.Extensions, len(cfg.NodeTypes))
	}
	t.Logf("ExtensionMap has %d entries", len(extensionMap))
	for ext, cfg := range extensionMap {
		t.Logf("  - %s -> %s", ext, cfg.Name)
	}
}
