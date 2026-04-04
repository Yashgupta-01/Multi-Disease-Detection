async function predict() {
    const fileInput = document.getElementById("fileInput");
    const disease = document.getElementById("disease").value;
    const resultDiv = document.getElementById("result");

    if (!fileInput.files.length) {
        alert("Please upload an image");
        return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    resultDiv.innerHTML = "⏳ Predicting...";

    try {
        const response = await fetch(`http://127.0.0.1:8000/predict/${disease}`, {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        resultDiv.innerHTML = `
            <h3>🧠 Disease: ${data.disease}</h3>
            <h3>📊 Prediction: ${data.prediction}</h3>
            <h3>✅ Confidence: ${(data.confidence * 100).toFixed(2)}%</h3>
        `;

    } catch (error) {
        resultDiv.innerHTML = "❌ Error connecting to backend";
        console.error(error);
    }
}