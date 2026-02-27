async function analyzeNews() {
    const inputBox = document.getElementById("newsInput");
    const resultBox = document.getElementById("result");

    const text = inputBox.value.trim();

    if (!text) {
        resultBox.innerHTML = "<p style='color: orange;'>⚠️ Please enter some news text first.</p>";
        return;
    }

    // Show loading state
    resultBox.innerHTML = "<p>🔍 Analyzing with BERT model...</p>";

    try {
        const response = await fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                text: text
            })
        });

        if (!response.ok) {
            throw new Error("API not responding");
        }

        const data = await response.json();

        // Dynamic color based on prediction
        const color = data.label === "FAKE" ? "#ff4d4d" : "#00ff99";

        resultBox.innerHTML = `
            <div style="font-size: 20px; font-weight: 600;">
                Prediction: <span style="color:${color}">${data.label}</span>
            </div>
            <div style="margin-top:10px;">
                Confidence: <b>${data.confidence}%</b>
            </div>
            <div>Fake Probability: ${data.fake_probability}%</div>
            <div>Real Probability: ${data.real_probability}%</div>
        `;
    } catch (error) {
        resultBox.innerHTML = `
            <p style="color:red;">
                ❌ Error: Cannot connect to API.<br>
                Make sure FastAPI server is running on port 8000.
            </p>
        `;
        console.error(error);
    }
}