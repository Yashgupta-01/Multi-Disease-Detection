// Image Upload Preview Logic
const fileInput = document.getElementById("fileInput");
const imagePreview = document.getElementById("imagePreview");
const uploadPlaceholder = document.getElementById("uploadPlaceholder");

fileInput.addEventListener("change", function () {
    const file = this.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function (e) {
            imagePreview.src = e.target.result;
            imagePreview.classList.remove("hidden");
            uploadPlaceholder.classList.add("hidden");
        };
        reader.readAsDataURL(file);
    } else {
        imagePreview.src = "";
        imagePreview.classList.add("hidden");
        uploadPlaceholder.classList.remove("hidden");
    }
});

async function predict() {
    const disease = document.getElementById("disease").value;
    const resultDiv = document.getElementById("result");
    const loadingDiv = document.getElementById("loading");
    const predictBtn = document.getElementById("predictBtn");

    if (!fileInput.files.length) {
        alert("⚠️ Please upload a medical image first.");
        return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    // UI State: Loading
    resultDiv.classList.add("hidden");
    loadingDiv.classList.remove("hidden");
    predictBtn.disabled = true;
    predictBtn.style.opacity = "0.7";

    try {
        const response = await fetch(`http://127.0.0.1:8000/predict/${disease}`, {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.detail || "Server error");
        }

        const data = await response.json();

        // Determine color based on outcome (assuming 'Normal' or 'Benign' is good)
        let isAlert = true;
        const predStr = data.prediction.toLowerCase();
        if (predStr.includes("normal") || predStr.includes("benign") || predStr.includes("no tumor")) {
            isAlert = false;
        }

        const colorVar = isAlert ? "var(--warning)" : "var(--primary)";

        // Build premium result HTML
        resultDiv.innerHTML = `
            <div class="result-header">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>
                Analysis Complete - ${data.disease}
            </div>
            <div class="prediction-value" style="color: ${colorVar}">
                ${data.prediction}
            </div>
            <div class="confidence-wrapper">
                <div class="confidence-header">
                    <span>AI Confidence</span>
                    <span>${data.confidence.toFixed(2)}%</span>
                </div>
                <div class="confidence-bar-bg">
                    <div class="confidence-bar-fill" style="width: 0%; background: ${colorVar}"></div>
                </div>
            </div>
        `;

        // UI State: Done
        loadingDiv.classList.add("hidden");
        resultDiv.classList.remove("hidden");

        // Animate the progress bar
        setTimeout(() => {
            const bar = resultDiv.querySelector('.confidence-bar-fill');
            if (bar) bar.style.width = `${data.confidence}%`;
        }, 50);

    } catch (error) {
        resultDiv.innerHTML = `
            <div style="color: var(--danger); text-align: center;">
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="margin-bottom: 10px"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>
                <p><strong>Error linking to backend:</strong></p>
                <p style="font-size: 0.9em; margin-top: 5px">${error.message}</p>
            </div>
        `;
        loadingDiv.classList.add("hidden");
        resultDiv.classList.remove("hidden");
        console.error(error);
    } finally {
        predictBtn.disabled = false;
        predictBtn.style.opacity = "1";
    }
}