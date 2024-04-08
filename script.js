document.getElementById("send-btn").addEventListener("click", sendMessage);

document
  .getElementById("user-input")
  .addEventListener("keypress", function (e) {
    if (e.key === "Enter") {
      e.preventDefault(); // Prevents the default action
      sendMessage(); // Calls the function to send the message
    }
  });

function sendMessage() {
  const userInput = document.getElementById("user-input");
  const userText = userInput.value.trim();
  if (userText !== "") {
    displayMessage(userText, "user");
    fetch("http://127.0.0.1:5000/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ message: userText }),
    })
      .then((response) => response.json())
      .then((data) => {
        displayMessage(data.response, "bot");
      })
      .catch((error) => {
        console.error("Error:", error);
        displayMessage(
          "Sorry, I can't connect to the chat service right now.",
          "bot"
        );
      });

    userInput.value = ""; // Clear input after sending
  }
}

function displayMessage(message, sender) {
  const chatBox = document.getElementById("chat-box");
  const messageDiv = document.createElement("div");
  messageDiv.textContent = message;
  messageDiv.classList.add(sender); // Adds 'user' or 'bot' class for styling
  chatBox.appendChild(messageDiv);
  chatBox.scrollTop = chatBox.scrollHeight; // Auto-scrolls to the bottom
}
