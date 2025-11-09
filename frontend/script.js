document.addEventListener("DOMContentLoaded", () => {
    const messages = document.getElementById("messages");
    const userInput = document.getElementById("userInput");
    const sendButton = document.getElementById("sendButton");

    let userId = localStorage.getItem("userId");
    if (!userId) {
        userId = "user_" + Math.random().toString(36).substr(2, 9);
        localStorage.setItem("userId", userId);
    }

    const addMessage = (text, sender) => {
        const message = document.createElement("div");
        message.classList.add("message", sender);
        message.innerText = text;
        messages.appendChild(message);
        messages.scrollTop = messages.scrollHeight;
    };

    const sendMessage = async () => {
        const text = userInput.value.trim();
        if (text === "") return;

        addMessage(text, "user");
        userInput.value = "";

        try {
            const response = await fetch("/query", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ user_id: userId, text: text }),
            });

            const data = await response.json();
            addMessage(data.answer, "bot");
        } catch (error) {
            console.error("Error:", error);
            addMessage("Произошла ошибка при отправке запроса.", "bot");
        }
    };

    sendButton.addEventListener("click", sendMessage);
    userInput.addEventListener("keydown", (event) => {
        if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            sendMessage();
        }
    });
});
