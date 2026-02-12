const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const captureBtn = document.getElementById("captureBtn");
const result = document.querySelector(".result");
const loading = document.querySelector(".loading");
const error = document.querySelector(".error");

// Start camera
async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment" }, // Use back camera on mobile
    });
    video.srcObject = stream;
    captureBtn.disabled = false;
  } catch (err) {
    showError("Could not access camera. Please allow camera permissions.");
    console.error("Camera error:", err);
  }
}

// Capture and predict
captureBtn.addEventListener("click", async () => {
  // Hide previous results
  result.classList.remove("show");
  error.classList.remove("show");
  loading.classList.add("show");
  captureBtn.disabled = true;

  // Capture frame
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext("2d").drawImage(video, 0, 0);

  // Convert to base64
  const imageData = canvas.toDataURL("image/jpeg");

  try {
    // Send to server
    const response = await fetch("/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ image: imageData }),
    });

    const data = await response.json();

    if (data.error) {
      throw new Error(data.error);
    }

    // Display results
    displayResults(data);
  } catch (err) {
    showError("Prediction failed: " + err.message);
    console.error("Prediction error:", err);
  } finally {
    loading.classList.remove("show");
    captureBtn.disabled = false;
  }
});

function displayResults(data) {
  // Update category
  const categoryEl = document.getElementById("category");
  categoryEl.textContent = data.category;
  categoryEl.className = "category " + data.category;

  // Update confidence
  document.getElementById("confidence").textContent =
    `${(data.confidence * 100).toFixed(1)}% confidence`;

  // Update bars
  const barsEl = document.getElementById("bars");
  barsEl.innerHTML = "";

  const categories = ["compost", "recycle", "landfill"];
  categories.forEach((cat) => {
    const value = data.all_predictions[cat];
    const percentage = (value * 100).toFixed(1);

    const barItem = document.createElement("div");
    barItem.className = "bar-item";
    barItem.innerHTML = `
            <div class="bar-label">
                <span>${cat.charAt(0).toUpperCase() + cat.slice(1)}</span>
                <span>${percentage}%</span>
            </div>
            <div class="bar-bg">
                <div class="bar-fill ${cat}" style="width: ${percentage}%">
                </div>
            </div>
        `;
    barsEl.appendChild(barItem);
  });

  // Show result
  result.classList.add("show");
}

function showError(message) {
  document.getElementById("errorMessage").textContent = message;
  error.classList.add("show");
}

// Start camera on page load
startCamera();
