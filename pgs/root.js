const fileInput = document.getElementById("ancestryFile");
const insightsContainer = document.getElementById("insightsContainer");
const returnBtn = document.getElementById("returnBtn");

// Sample insights data
const ancestryInsights = {
  "Europe": "Rich medieval history with diverse cultures from the Renaissance to modern times.",
  "Asia": "Ancient civilizations with strong traditions in philosophy, art, and technology.",
  "Africa": "Diverse tribal cultures and kingdoms with deep historical roots.",
  "Americas": "Mixture of Indigenous civilizations and colonial influences."
};

// Return button functionality
returnBtn.addEventListener("click", () => {
  window.history.back(); // Go back to previous page
});

// Handle file upload
fileInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = function(event) {
    let content = event.target.result;
    let regions = [];

    try {
      if (file.name.endsWith(".json")) {
        const data = JSON.parse(content);
        regions = data.regions || [];
      } else if (file.name.endsWith(".csv")) {
        const lines = content.split("\n");
        regions = lines.map(line => line.trim()).filter(line => line);
      }

      displayInsights(regions);
    } catch (err) {
      alert("Error reading file: " + err.message);
    }
  };
  reader.readAsText(file);
});

function displayInsights(regions) {
  insightsContainer.innerHTML = "";

  if (!regions.length) {
    insightsContainer.innerHTML = `<div class="insight-card">No regions found in the file.</div>`;
    return;
  }

  regions.forEach(region => {
    const insightText = ancestryInsights[region] || "Cultural/historical insights not available.";
    const card = document.createElement("div");
    card.className = "insight-card";
    card.textContent = `${region}: ${insightText}`;
    insightsContainer.appendChild(card);
  });
}
