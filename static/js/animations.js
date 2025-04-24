document.addEventListener("DOMContentLoaded", () => {
    // Fade in elements on load
    document.querySelectorAll(".step-info, .persona-card, .debate-message").forEach(el => {
        el.style.opacity = 0;
        setTimeout(() => {
            el.style.transition = "opacity 0.7s ease-in";
            el.style.opacity = 1;
        }, 200);
    });
    // Animate icons on hover
    document.querySelectorAll(".step-icon").forEach(icon => {
        icon.addEventListener("mouseover", () => {
            icon.style.transform = "scale(1.3)";
        });
        icon.addEventListener("mouseout", () => {
            icon.style.transform = "scale(1)";
        });
    });
});
