// Keep track of all chats
let chats = [];
let activeChatId = -1;
let active_embedding_function = "";
let active_vector_database = "";
    
const chatBox = document.getElementById("chat-box");
const userInput = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");
const chatList = document.getElementById("chat-list");
const newChatBtn = document.getElementById("new-chat");

const k_value_input = document.getElementById("k-value");

const config = document.getElementById("chatConfigModal");
const config_createBtn = document.getElementById("createChatBtn");
const config_cancelBtn = document.getElementById("cancelChatBtn");


// initialize the chatbot 
document.addEventListener("DOMContentLoaded", () => {
    fetch("/data")
        .then(response => response.json())
        .then(data => {
            if (!data || data.length === 0) {
                console.log("No data found, skipping...");
                return; // exit early
            }

            data.forEach(r => {
                chats.push({
                    id: chats.length + 1,
                    embedding_function: r.embeddings,
                    vector_database: r.db,
                    messages: []
                });
            });

            // set the first chat default chat
            activeChatId = 1;
            renderChatList();
            renderMessages();
            switchRetriever();

        })
        .catch(err => {
            console.error("Error fetching data:", err);
        });
});


// Render chat sessions in sidebar
function renderChatList() {
    chatList.innerHTML = "";

    if (!chats || chats.length === 0) {
        console.log("No chats to render.");
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

    chatBox.innerHTML = "";

    if (activeChat.messages.length === 0) {
        chatBox.innerHTML = `
      <div class="message bot-message">
        <pre>Hello! Ask me anything, and I'll do my best to answer.</pre>
      </div>`;
    } else {
        activeChat.messages.forEach(msg => {
            const msgDiv = document.createElement("div");
            msgDiv.classList.add("message");
            msgDiv.classList.add(msg.sender === "user" ? "user-message" : "bot-message");
            msgDiv.innerHTML = `<pre>${msg.text}</pre>`;
            chatBox.appendChild(msgDiv);
        });
    }

    chatBox.scrollTop = chatBox.scrollHeight;
}


function switchRetriever() {

    fetch("/switch", {
            method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ db: active_vector_database, embeddings: active_embedding_function, k_value: k_value_input.value.trim() })
    })
        .then(response => response.json())
        .then(data => {
            console.log("Backend updated:", data.ok);
        })
        .catch(error => {
            console.error("Error updating backend:", error);
        });
}

// Send message
function sendMessage() {
    const text = userInput.value.trim();
    if (!text) return;

    const activeChat = chats.find(c => c.id === activeChatId);

    // Add user message
    activeChat.messages.push({ sender: "user", text: text });
    userInput.value = "";
    renderMessages();

    // Add temporary bot message
    const typingMsg = { sender: "bot", text: "The agent is writing..." };
    activeChat.messages.push(typingMsg);
    renderMessages();

    // Send request to Flask backend
    fetch("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: text })
    })
        .then(response => response.json())
        .then(data => {
            // Replace typing message with real answer
            const index = activeChat.messages.indexOf(typingMsg);
            if (index !== -1) {
                activeChat.messages[index].text = data.answer;
            } else {
                activeChat.messages.push({ sender: "bot", text: data.answer });
            }
            renderMessages();
        })
        .catch(error => {
            const index = activeChat.messages.indexOf(typingMsg);
            if (index !== -1) {
                activeChat.messages[index].text = "Error: " + error.message;
            } else {
                activeChat.messages.push({ sender: "bot", text: "Error: " + error.message });
            }
            renderMessages();
        });
}


// Create a new chat
newChatBtn.addEventListener("click", () => {
    config.style.display = "flex";
});


config_createBtn.addEventListener("click", () => {
    const embedFunc = document.getElementById("embedInput").value;
    const vectorDB = document.getElementById("vectorInput").value;
    const file = document.getElementById("csvFileInput").files[0];

    const newId = Date.now();

    if (!file) {
        alert("Please select a CSV file first.");
        return;
    }

    const formData = new FormData();
    formData.append("embeddings", embedFunc);
    formData.append("db", vectorDB);
    formData.append("csv_file", file); 

    fetch("/new", {
        method: "POST",
        body: formData
    })
    .then(res => res.json())
    .then(data => {
        console.log("New chat created:", data);
        chats.push({
            id: newId,
            embedding_function: embedFunc,
            vector_database: vectorDB,
            messages: []
        });

        activeChatId = newId;

        config.style.display = "none";

        renderChatList();
        renderMessages();
        switchRetriever();

    })
    .catch(err => console.error("Error:", err));
});


config_cancelBtn.addEventListener("click", () => {
    config.style.display = "none";
});


// Event listeners
sendBtn.addEventListener("click", sendMessage);
userInput.addEventListener("keypress", e => {
    if (e.key === "Enter") sendMessage();
});




