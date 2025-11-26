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

func TestTreeSitter_SyntaxError(t *testing.T) {
	// Python with syntax error should fall back to size-based
	content := `
def broken(:
    pass
`

	chunks, err := ChunkFile("/test/broken.py", content, nil)
	if err != nil {
		t.Fatalf("ChunkFile failed: %v", err)
	}

	// Should still produce chunks via fallback
	if len(chunks) == 0 {
		t.Error("expected fallback to size-based chunking for syntax errors")
	}
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

