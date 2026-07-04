// State management
let chats = [];
let activeChatId = -1;
let active_embedding_function = "";
let active_vector_database = "";
let active_generator_model = "qwen2.5:3b";

// DOM Elements
const chatBox = document.getElementById("chat-box");
const userInput = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");
const chatList = document.getElementById("chat-list");
const newChatBtn = document.getElementById("new-chat");

// Sliders and Displays
const k_value_input = document.getElementById("k-value");
const k_val_display = document.getElementById("k-val-display");
const n_value_input = document.getElementById("n-value");
const n_val_display = document.getElementById("n-val-display");

// Dashboard display elements
const active_gen_display = document.getElementById("active-gen-display");
const active_embed_display = document.getElementById("active-embed-display");
const p_k_val = document.getElementById("p-k-val");
const p_n_val = document.getElementById("p-n-val");

// Configuration Modal Elements
const config = document.getElementById("chatConfigModal");
const config_createBtn = document.getElementById("createChatBtn");
const config_cancelBtn = document.getElementById("cancelChatBtn");

// Initialize chatbot sessions
document.addEventListener("DOMContentLoaded", () => {
    fetch("/data")
        .then(response => response.json())
        .then(data => {
            if (!data || data.length === 0) {
                console.log("No databases found. Waiting for automatic demo index...");
                // Reload list in 3 seconds to check if demo dataset finished indexing
                setTimeout(reloadSessions, 4000);
                return;
            }

            data.forEach(r => {
                chats.push({
                    id: chats.length + 1,
                    embedding_function: r.embeddings,
                    vector_database: r.db,
                    generator_model: r.generator || "qwen2.5:3b",
                    messages: []
                });
            });

            // Set the first chat as default active chat
            activeChatId = 1;
            renderChatList();
            renderMessages();
            switchRetriever();
        })
        .catch(err => {
            console.error("Error fetching sessions data:", err);
        });
});

// Reload session data after startup indexing
function reloadSessions() {
    fetch("/data")
        .then(response => response.json())
        .then(data => {
            if (data && data.length > 0) {
                chats = [];
                data.forEach(r => {
                    chats.push({
                        id: chats.length + 1,
                        embedding_function: r.embeddings,
                        vector_database: r.db,
                        generator_model: r.generator || "qwen2.5:3b",
                        messages: []
                    });
                });
                activeChatId = 1;
                renderChatList();
                renderMessages();
                switchRetriever();
            }
        });
}

// Render sessions in the sidebar
function renderChatList() {
    chatList.innerHTML = "";

    if (!chats || chats.length === 0) {
        chatList.innerHTML = `<div style="padding: 1rem; font-size: 0.8rem; color: #94A3B8;">No databases loaded yet. Click create new below.</div>`;
        return;
    }

    chats.forEach(chat => {
        const div = document.createElement("div");
        div.classList.add("chat-session");
        if (chat.id === activeChatId) div.classList.add("active");
        div.textContent = chat.vector_database;
        div.onclick = () => {
            activeChatId = chat.id;
            renderChatList();
            renderMessages();
            switchRetriever();
        };
        chatList.appendChild(div);
    });
}

// Render messages of the active chat
function renderMessages() {
    if (!chats || chats.length === 0) {
        return;
    }

    const activeChat = chats.find(c => c.id === activeChatId);
    active_embedding_function = activeChat.embedding_function;
    active_vector_database = activeChat.vector_database;
    active_generator_model = activeChat.generator_model || "qwen2.5:3b";

    // Update top dashboard badges
    if (active_gen_display) active_gen_display.textContent = active_generator_model;
    if (active_embed_display) active_embed_display.textContent = active_embedding_function;
    if (p_k_val) p_k_val.textContent = k_value_input.value;
    if (p_n_val) p_n_val.textContent = n_value_input.value;

    chatBox.innerHTML = "";

    if (activeChat.messages.length === 0) {
        chatBox.innerHTML = `
            <div class="message bot-message">
                <div class="message-text">Hello! Ask me any academic question, and I'll retrieve research papers from the <strong>${active_vector_database}</strong> database using the <strong>${active_embedding_function}</strong> retriever to construct a grounded response.</div>
            </div>`;
    } else {
        activeChat.messages.forEach(msg => {
            const msgDiv = document.createElement("div");
            msgDiv.classList.add("message");
            msgDiv.classList.add(msg.sender === "user" ? "user-message" : "bot-message");
            
            // Render text
            const textDiv = document.createElement("div");
            textDiv.classList.add("message-text");
            textDiv.textContent = msg.text;
            msgDiv.appendChild(textDiv);

            // Render sources accordion if bot message contains references
            if (msg.sender === "bot" && msg.sources && msg.sources.length > 0) {
                const accordionDiv = document.createElement("div");
                accordionDiv.classList.add("sources-accordion");
                
                const header = document.createElement("div");
                header.classList.add("sources-header");
                header.innerHTML = `<span>📂 View Retrieved Sources (${msg.sources.length} papers)</span><span class="sources-icon">▼</span>`;
                
                const content = document.createElement("div");
                content.classList.add("sources-content");
                
                msg.sources.forEach(src => {
                    const item = document.createElement("div");
                    item.classList.add("source-item");
                    item.innerHTML = `
                        <div class="source-title">${src.title}</div>
                        <div class="source-snippet">${src.content.replace(/\n/g, '<br>')}</div>
                    `;
                    content.appendChild(item);
                });
                
                header.onclick = () => {
                    accordionDiv.classList.toggle("open");
                };
                
                accordionDiv.appendChild(header);
                accordionDiv.appendChild(content);
                msgDiv.appendChild(accordionDiv);
            }

            chatBox.appendChild(msgDiv);
        });
    }

    chatBox.scrollTop = chatBox.scrollHeight;
}

// Switch database or parameters in backend
function switchRetriever() {
    fetch("/switch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
            db: active_vector_database, 
            embeddings: active_embedding_function, 
            generator: active_generator_model,
            k_value: k_value_input.value.trim(),
            n_value: n_value_input.value.trim()
        })
    })
    .then(response => response.json())
    .then(data => {
        console.log("RAG configuration successfully updated on server:", data.ok);
    })
    .catch(error => {
        console.error("Error switching RAG configuration:", error);
    });
}

// Send user query
function sendMessage() {
    const text = userInput.value.trim();
    if (!text) return;

    const activeChat = chats.find(c => c.id === activeChatId);

    // Add user message
    activeChat.messages.push({ sender: "user", text: text });
    userInput.value = "";
    renderMessages();

    // Disable input interface and show loading spinner
    userInput.disabled = true;
    sendBtn.disabled = true;
    sendBtn.innerHTML = `<span>Analyzing...</span>`;

    // Add temporary typing placeholder
    const typingMsg = { sender: "bot", text: "Retrieving literature and generating answer..." };
    activeChat.messages.push(typingMsg);
    renderMessages();

    // Send query request to backend
    fetch("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: text })
    })
    .then(response => response.json())
    .then(data => {
        // Re-enable input interface
        userInput.disabled = false;
        sendBtn.disabled = false;
        sendBtn.innerHTML = `<span>Send Query</span>`;

        // Replace placeholder message with final response and references
        const index = activeChat.messages.indexOf(typingMsg);
        if (index !== -1) {
            activeChat.messages[index].text = data.answer;
            activeChat.messages[index].sources = data.sources || [];
        } else {
            activeChat.messages.push({ sender: "bot", text: data.answer, sources: data.sources || [] });
        }
        renderMessages();
        userInput.focus();
    })
    .catch(error => {
        userInput.disabled = false;
        sendBtn.disabled = false;
        sendBtn.innerHTML = `<span>Send Query</span>`;

        const index = activeChat.messages.indexOf(typingMsg);
        const errMsg = "Server connection lost. Please verify local models are running in Ollama.";
        if (index !== -1) {
            activeChat.messages[index].text = errMsg;
        } else {
            activeChat.messages.push({ sender: "bot", text: errMsg });
        }
        renderMessages();
    });
}

// Modal open/close actions
newChatBtn.addEventListener("click", () => {
    config.style.display = "flex";
});

config_cancelBtn.addEventListener("click", () => {
    config.style.display = "none";
});

// Create new RAG index session
config_createBtn.addEventListener("click", () => {
    const embedFunc = document.getElementById("embedInput").value;
    const vectorDB = document.getElementById("vectorInput").value.trim().replace(/\s+/g, "_");
    const generatorModel = document.getElementById("generatorInput").value;
    const file = document.getElementById("csvFileInput").files[0];

    if (!vectorDB) {
        alert("Please enter a database name.");
        return;
    }
    if (!file) {
        alert("Please select a CSV file first.");
        return;
    }

    // Show indexing loader status
    config_createBtn.disabled = true;
    config_createBtn.textContent = "Indexing vectors...";

    const formData = new FormData();
    formData.append("embeddings", embedFunc);
    formData.append("db", vectorDB);
    formData.append("generator", generatorModel);
    formData.append("csv_file", file); 

    fetch("/new", {
        method: "POST",
        body: formData
    })
    .then(res => res.json())
    .then(data => {
        config_createBtn.disabled = false;
        config_createBtn.textContent = "Index & Create";
        config.style.display = "none";

        if (data.error) {
            alert("Error indexing database: " + data.error);
            return;
        }

        const newId = Date.now();
        chats.push({
            id: newId,
            embedding_function: embedFunc,
            vector_database: vectorDB,
            generator_model: generatorModel,
            messages: []
        });

        activeChatId = newId;
        renderChatList();
        renderMessages();
        switchRetriever();
    })
    .catch(err => {
        config_createBtn.disabled = false;
        config_createBtn.textContent = "Index & Create";
        console.error("Error creating new index:", err);
        alert("An error occurred during indexing. Verify Ollama embedding model is pulled.");
    });
});

// Slider values listeners
k_value_input.addEventListener("input", () => {
    k_val_display.textContent = k_value_input.value;
    if (p_k_val) p_k_val.textContent = k_value_input.value;
});

k_value_input.addEventListener("change", () => {
    // Force rerank slider constraint: Rerank N cannot exceed Retriever pool K
    if (parseInt(n_value_input.value) > parseInt(k_value_input.value)) {
        n_value_input.value = k_value_input.value;
        n_val_display.textContent = k_value_input.value;
        if (p_n_val) p_n_val.textContent = k_value_input.value;
    }
    switchRetriever();
});

n_value_input.addEventListener("input", () => {
    // Constraint check
    if (parseInt(n_value_input.value) > parseInt(k_value_input.value)) {
        n_value_input.value = k_value_input.value;
    }
    n_val_display.textContent = n_value_input.value;
    if (p_n_val) p_n_val.textContent = n_value_input.value;
});

n_value_input.addEventListener("change", () => {
    switchRetriever();
});

// Message send events
sendBtn.addEventListener("click", sendMessage);
userInput.addEventListener("keypress", e => {
    if (e.key === "Enter") sendMessage();
});
