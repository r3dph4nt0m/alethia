const flashcardData = {
  spanish: [
    { front: "Hola", back: "Hello" },
    { front: "Adiós", back: "Goodbye" },
    { front: "Gracias", back: "Thank you" },
    { front: "Por favor", back: "Please" }
  ],
  french: [
    { front: "Bonjour", back: "Hello" },
    { front: "Au revoir", back: "Goodbye" },
    { front: "Merci", back: "Thank you" },
    { front: "S'il vous plaît", back: "Please" }
  ],
  german: [
    { front: "Hallo", back: "Hello" },
    { front: "Tschüss", back: "Goodbye" },
    { front: "Danke", back: "Thank you" },
    { front: "Bitte", back: "Please" }
  ]
};

let currentIndex = 0;
let language = "spanish";

const flashcardEl = document.getElementById("flashcard");
const selectEl = document.getElementById("language");

function showCard() {
  const card = flashcardData[language][currentIndex];
  flashcardEl.querySelector(".front").textContent = card.front;
  flashcardEl.querySelector(".back").textContent = card.back;
  flashcardEl.classList.remove("flipped");
}

// Flip card
flashcardEl.addEventListener("click", () => {
  flashcardEl.classList.toggle("flipped");
});

// Navigation
document.getElementById("next").addEventListener("click", () => {
  currentIndex = (currentIndex + 1) % flashcardData[language].length;
  showCard();
});
document.getElementById("prev").addEventListener("click", () => {
  currentIndex = (currentIndex - 1 + flashcardData[language].length) % flashcardData[language].length;
  showCard();
});

// Language change
selectEl.addEventListener("change", (e) => {
  language = e.target.value;
  currentIndex = 0;
  showCard();
});

// Initialize
showCard();
