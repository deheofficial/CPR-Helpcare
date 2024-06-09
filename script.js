function nextStep(step) {
    const steps = document.querySelectorAll('.step');
    steps.forEach((el) => (el.style.display = 'none'));
    document.getElementById(`step-${step}`).style.display = 'block';
}
