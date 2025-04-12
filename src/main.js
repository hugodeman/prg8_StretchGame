import { PoseLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18";

ml5.setBackend("webgl");
const nn = ml5.neuralNetwork({
    task: 'classification',
    debug: true,
})

const options = {
    model: "./model.json",
    metadata : "./model_meta.json",
    weights:"./model.weights.bin"
}

const poseImages = {
    'left': './public/armStretchLeft.png',
    'right': './public/armStretchRight.png',
    'up': './public/armsStretchUp.png'
}

const poses = [
    { label: "left" },
    { label: "right" },
    { label: "up" }
];

let currentPose = '';
let previousPose = null;
let currentPoseStartTime = null;
let score = 0;

const scoreSpan = document.getElementById('score')
const poseInstruction = document.getElementById("poseInstruction");

const webCamButton = document.getElementById('webcamButton')
const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");

let poseLandmarker = undefined;
let webcamRunning = false;

webCamButton.addEventListener("click", () => {
    createPoseLandmarker();
});

function getRandomPose() {
    const availablePoses = poses.filter(p => p.label !== previousPose?.label);

    const selectedPose = availablePoses[Math.floor(Math.random() * availablePoses.length)];

    previousPose = selectedPose;
    updatePoseImage(selectedPose.label);
    return selectedPose;
}

function updatePoseImage(poseLabel) {
    const imageElement = document.getElementById("poseImage");

    if (poseLabel === "rest" || !poseImages[poseLabel]) {
        imageElement.style.display = "none";
    } else {
        imageElement.src = poseImages[poseLabel];
        imageElement.style.display = "block";
    }
}

async function createPoseLandmarker() {
    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm");
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task`,
            delegate: "GPU"
        },
        runningMode: "VIDEO",
        numPoses: 1
    });
    enableCam()
}

async function enableCam() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        video.srcObject = stream;
        webCamButton.style.display = 'none';
        scoreSpan.innerText= 'score: ' + score;
        video.addEventListener("loadeddata", () => {
            canvasElement.width = video.videoWidth;
            canvasElement.height = video.videoHeight;
            nn.load(options, () => {
                webcamRunning = true
                currentPose = getRandomPose();
                poseInstruction.innerText = "Doe deze pose: " + currentPose.label;
                predictWebcam();
            });
        });
    } catch (error) {
        console.error("Error accessing webcam:", error);
    }
}

async function predictWebcam() {
    const results = await poseLandmarker.detectForVideo(video, performance.now());
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

    if (results.landmarks.length > 0) {
        const input = results.landmarks[0].flatMap(p => [p.x, p.y, p.z]);

        if (input && input.length > 0) {
            nn.classify(input, (results) => {
                if (results && results[0]) {
                    const detectedPose = results[0].label;
                    console.log(`Detected pose: ${detectedPose}`);

                    if (detectedPose === currentPose.label) {
                        if (currentPoseStartTime === null) {
                            currentPoseStartTime = performance.now();
                        } else {
                            const elapsedTime = (performance.now() - currentPoseStartTime) / 1000;
                            if (elapsedTime >= 3) {
                                score += 1;
                                scoreSpan.innerText = 'score: ' + score;
                                currentPoseStartTime = null;
                                currentPose = getRandomPose();
                                poseInstruction.innerText = "Doe deze pose: " + currentPose.label;
                            }
                        }
                    } else {
                        currentPoseStartTime = null;
                    }
                }
            });
        }
    }
    requestAnimationFrame(predictWebcam);
}

