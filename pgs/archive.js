const tabs = document.querySelectorAll(".tab-btn");
const panels = document.querySelectorAll(".tab-panel");

tabs.forEach(tab => {
  tab.addEventListener("click", () => {
    // Remove active class from all tabs and panels
    tabs.forEach(t => t.classList.remove("active"));
    panels.forEach(p => p.classList.remove("active"));

    // Add active to clicked tab and corresponding panel
    tab.classList.add("active");
    document.getElementById(tab.dataset.tab).classList.add("active");
  });
});
