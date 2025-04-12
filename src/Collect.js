import { PoseLandmarker, FilesetResolver, DrawingUtils } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18";

const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
let webcamRunning = false;

let poseLandmarker = null;
let drawUtils = null;
let results = null;
let collectedData = [];

let isTraining = false;
let trainingStartTime = null;
let trainingDuration = 100;
let currentPoseLabel = null;

let cooldownTime = 100;
let lastTrainingTime = 0;

let poseDataCount = {
    'left': 0,
    'right': 0,
    'up': 0,
    'rest': 0
};

const MIN_DATASETS = 25;

document.getElementById("webcamButton").addEventListener("click", createPoseLandmarker);

document.getElementById("saveButton").addEventListener("click", () => {
    const blob = new Blob([JSON.stringify(collectedData)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "pose-data.json";
    a.click();
});

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
    drawUtils = new DrawingUtils(canvasCtx);
    enableCam()
}

async function enableCam() {
    webcamRunning = true
    try {
        const stream = await navigator.mediaDevices.getUserMedia({video: true});
        video.srcObject = stream;

        video.addEventListener("loadeddata", () => {
            canvasElement.width = video.videoWidth;
            canvasElement.height = video.videoHeight;
            predictWebcam();
        });
    } catch (error) {
        console.error("Error accessing webcam:", error);
    }
}

function storeData(label) {
    const flatLandmarks = results.landmarks[0].flatMap(point => [point.x, point.y, point.z]);

    if (poseDataCount[label] < MIN_DATASETS) {
        collectedData.push({
            points: flatLandmarks,
            label: label
        });

        poseDataCount[label]++;
        console.log(`Saved data for ${label}. Total for ${label}: ${poseDataCount[label]}`);
    }

    if (poseDataCount[label] >= MIN_DATASETS && !isTraining) {
        console.log(`Start training voor pose: ${label}`);
        startTraining(label);
    }
}

async function predictWebcam() {
    results = await poseLandmarker.detectForVideo(video, performance.now());
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    if (results.landmarks && results.landmarks.length > 0) {
        drawUtils.drawLandmarks(results.landmarks[0]);

        const poseDetected = detectPose(results.landmarks);
        const now = Date.now();
        if (
            poseDetected &&
            poseDetected !== "rest" &&
            !isTraining &&
            now - lastTrainingTime > cooldownTime
        ) {
            startTraining(poseDetected);
            lastTrainingTime = now;
        }

        if (isTraining && Date.now() - trainingStartTime > trainingDuration) {
            stopTraining();
        }

        document.addEventListener("keydown", (event) => {
            if (event.key === " " && poseDataCount["rest"] < MIN_DATASETS) {
                console.log("Training voor 'rest' gestart");
                storeData("rest");
            }
        });
    } else {
        console.log("Geen landmarks gedetecteerd");
    }
    requestAnimationFrame(predictWebcam);
}

function detectPose(landmarks) {
    const lm = landmarks[0];
    if (!lm || lm.length < 17) return null;

    const leftArmUp = lm[15].y < lm[11].y - 0.2;
    const rightArmUp = lm[16].y < lm[12].y - 0.2;
    const isBothArmsUp = leftArmUp && rightArmUp;

    if (leftArmUp && !rightArmUp) {
        return 'left';
    } else if (rightArmUp && !leftArmUp) {
        return 'right';
    } else if (isBothArmsUp) {
        return 'up';
    } else {
        return 'rest';
    }
}

function startTraining(pose) {
    const now = Date.now();
    if (now - lastTrainingTime < cooldownTime) {
        console.log("Cooldown actief, training nog niet opnieuw gestart.");
        return;
    }

    console.log("Training gestart voor pose:", pose);
    isTraining = true;
    currentPoseLabel = pose;
    trainingStartTime = now;
}

function stopTraining() {
    console.log("Training gestopt na tijdslimiet.");
    isTraining = false;
    lastTrainingTime = Date.now();

    if (currentPoseLabel) {
        storeData(currentPoseLabel);
        currentPoseLabel = null;
    }
}