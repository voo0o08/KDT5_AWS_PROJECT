const words = ['Codeit', 'JavaScript', 'DOM', 'document', 'window', 'Event', 'Bubbling', 'Delegation'];
const container = document.querySelector('#container');

function getRandomInt(minimum, maximum) {
  const min = Math.ceil(minimum);
  const max = Math.floor(maximum);
  
  return Math.floor(Math.random() * (max - min)) + min; 
}

function init() {
  const maxPositionX = container.offsetWidth - 90;
  const maxPositionY = container.offsetHeight - 100;
  
  for (let word of words) {
    const span = document.createElement('span');
    span.classList.add('word');
    span.style.top = `${getRandomInt(20, maxPositionY)}px`;
    span.style.left = `${getRandomInt(20, maxPositionX)}px`;
    span.dataset.word = word;
    span.textContent = word;
    container.append(span);
  }
}

init();
// initializer 끝 

const input = document.querySelector('#input');

function checker() {
  const words = document.querySelectorAll('.word');
  if (words.length === 0) {
    alert('Success!👏🏻');
    if(confirm('retry?')) {
      window.location.reload();
    }
  }
}

function removeWord() {
  const word = document.querySelector(`[data-word="${input.value}"]`);
  if (word) {
    word.remove();
    checker();
  }

  input.value = '';
}

input.addEventListener('change', removeWord);
