alert('게임을 시작합니다!👏🏻')
const words = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'];
// , 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
const container = document.querySelector('#container');

function getRandomInt(minimum, maximum) {
  const min = Math.ceil(minimum);
  const max = Math.floor(maximum);
  return Math.floor(Math.random() * (max - min)) + min; 
}

// 단어 초기화 
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
const input2 = document.querySelector('#captureButton');

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
  // `[data-word="${input.value}"]`; -> span태그의 data-word속성 중 input창에 입력한 값(input.value)와 일치하는 요소를 선택함.
  // ` -> 템플릿 리터럴
  // [] -> 속성 선택, 일치하는 모든 요소 선택해줌 
  const word = document.querySelector(`[data-word="${input.value}"]`);
  if (word) {
    word.remove();
    checker();
  }

  input.value = '';
}

input.addEventListener('change', removeWord);
// input2.addEventListener('click')

// 카메라 
// 사용자 얼굴을 표시할 비디오 요소 선택
const userFaceElement = document.getElementById('video');

// getUserMedia()를 사용하여 웹캠에 접근
navigator.mediaDevices.getUserMedia({ video: true })
  .then(function(stream) {
    // 비디오 요소의 소스에 스트림 연결
    userFaceElement.srcObject = stream;
    userFaceElement.play();
  })
  .catch(function(err) {
    console.error('웹캠에 접근할 수 없습니다.', err); //console에 출력됨 
  });

// 버튼을 클릭하면 파이썬 파일에 요청이 간다고?? ㅇㅅㅇ
document.getElementById("captureButton").addEventListener("click", function() {
  // 버튼 클릭 시 서버에 요청을 보냄
  fetch("/capture")
    .then(response => response.json())
    .then(data => {
      // 서버 응답 데이터에서 prediction 값 추출
      const prediction = data.prediction;
      // prediction 값과 일치하는 요소 삭제
      console.log(prediction)
      const word = document.querySelector(`[data-word="${prediction}"]`);
      if (word) {
        word.remove();
        checker();
      }
    })

    .catch(error => console.error(error));
});

