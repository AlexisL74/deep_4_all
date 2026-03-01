async function submitDescription() {
    const textarea = document.getElementById("description");
    const text = textarea.value.trim();
    if (!text) return;

    const resultDiv = document.getElementById("result");
    // clear last output
    resultDiv.innerHTML = "";

    textarea.value = "";

    const response = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text })
    });

    const data = await response.json();

    const out = document.createElement("div");
    out.className = "message bot";
    out.innerText = data.response;
    resultDiv.appendChild(out);
}