ml5.setBackend("webgl");
const nn = ml5.neuralNetwork(
    {
        task: 'classification',
        debug: true,
        layers: [
            {
                type: 'dense',
                units: 128,
                activation: 'relu',
            },
            {
                type: 'dense',
                units: 256,
                activation: 'relu',
            },
            {
                type: 'dense',
                activation: 'softmax'
            }
        ]
    }
)

let data = []
let trainData = []
let testData = []

document.getElementById("trainButton").addEventListener("click", async () => {
    await initData();
    await trainModel();
});

document.getElementById("testButton").addEventListener("click", async () => {
    await initData();
    await testModel();
});

function trainModel() {
    console.log('started training')

    if(data.length > 0){
        data.sort(() => (Math.random() - 0.5))

        trainData = data.slice(0, Math.floor(data.length * 0.8))
        testData = data.slice(Math.floor(data.length * 0.8) + 1)

        for (const {points, label} of trainData) {
            nn.addData(points, { label });
        }

        nn.normalizeData();
        nn.train({ epochs: 15 }, () => {
            console.log("Training done");
            nn.save();
        });
    }
}

async function testModel() {
    let correct = 0;

    for (const {points, label} of testData) {
        const prediction = await nn.classify(points);

        if (prediction[0].label === label){
            correct ++;
        }
    }
    console.log(`accuracy: ${correct / testData.length}`)
}

async function initData() {
    try {
        console.log('trying fetch')
        const response = await fetch("pose-data.json");
        const json = await response.json();

        data = json

        console.log("Data initialized:", data);
    } catch (error) {
        console.error("Fout bij laden van pose-data.json:", error);
    }
}
