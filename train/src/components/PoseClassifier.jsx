import { useRef, useState, useEffect } from "react";
import kNear from "../lib/kNear";
import { PoseLandmarker, FilesetResolver, DrawingUtils } from "@mediapipe/tasks-vision";

const DATA_URL = "/data/jerk_pose_data.json";
const MODEL_PATH = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task";
const WASM_PATH = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm";
const LEG_INDICES = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32];

// pose classifier component
export default function PoseClassifier() {
  const [modelTrained, setModelTrained] = useState(false);
  const [loading, setLoading] = useState(false);
  const [predictionLoading, setPredictionLoading] = useState(false);
  const [predictedStep, setPredictedStep] = useState("");
  const [error, setError] = useState("");
  const [accuracy, setAccuracy] = useState(null);
  const [confusionMatrix, setConfusionMatrix] = useState(null);
  const [allLabels, setAllLabels] = useState([]);
  const [numCorrect, setNumCorrect] = useState(0);
  const [numTest, setNumTest] = useState(0);
  const [evaluationDone, setEvaluationDone] = useState(false);
  const knnModelRef = useRef(null);
  const poseLandmarkerRef = useRef(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const animationRef = useRef(null);;


  // train knn model with json pose dataset
  const trainModel = async () => {
    setLoading(true);
    setError("");
    try {
      const res = await fetch(DATA_URL);
      const data = await res.json();
      const knn = new kNear(3);
      data.forEach((item) => knn.learn(item.data, item.label));
      knnModelRef.current = knn;
      setModelTrained(true);
    } catch (e) {
      setError("Failed to load or train model: " + e.message);
    }
    setLoading(false);
  };

  // start webcam & predict
  const handleStartWebcamAndPredict = async () => {
    setPredictionLoading(true);
    setError("");
    if (!modelTrained) {
      setError("Model not trained");
      setPredictionLoading(false);
      return;
    }
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
    }
    if (!poseLandmarkerRef.current) {
      const vision = await FilesetResolver.forVisionTasks(WASM_PATH);
      poseLandmarkerRef.current = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: MODEL_PATH, delegate: "GPU" },
        runningMode: "VIDEO",
        numPoses: 1,
      });
    }
    predictLoop();
    setPredictionLoading(false);
  };

  function getLegData(landmarks) {
    return LEG_INDICES.flatMap((idx) => [
      landmarks[idx].x,
      landmarks[idx].y,
    ]);
  }

  // prediction loop
  const predictLoop = async () => {
    if (
      !poseLandmarkerRef.current ||
      !videoRef.current ||
      !canvasRef.current ||
      !knnModelRef.current
    ) return;
    const video = videoRef.current;
    if (video.readyState === 4) {
      const width = video.videoWidth;
      const height = video.videoHeight;
      canvasRef.current.width = width;
      canvasRef.current.height = height;
      const now = performance.now();
      const result = await poseLandmarkerRef.current.detectForVideo(video, now);
      drawOnCanvas(result, width, height);
      if (result && result.landmarks && result.landmarks[0]) {
        const legData = getLegData(result.landmarks[0]);
        const step = knnModelRef.current.classify(legData);
        setPredictedStep(step);
      } else {
        setPredictedStep("");
      }
    }
    animationRef.current = requestAnimationFrame(predictLoop);
  };

  // draw on canvas
  function drawOnCanvas(result, width, height) {
    const ctx = canvasRef.current.getContext("2d");
    ctx.save();
    ctx.clearRect(0, 0, width, height);
    ctx.beginPath();
    ctx.moveTo(width / 2, 0);
    ctx.lineTo(width / 2, height);
    ctx.strokeStyle = "red";
    ctx.lineWidth = 4;
    ctx.stroke();

    if (result && result.landmarks) {
      const drawingUtils = new DrawingUtils(ctx);
      result.landmarks.forEach((landmarkSet) => {
        drawingUtils.drawLandmarks(landmarkSet, { radius: 6 });
        drawingUtils.drawConnectors(landmarkSet, PoseLandmarker.POSE_CONNECTIONS);
      });
    }
    ctx.restore();
  }

  // evaluate (accuracy & confusion matrix)
  const evaluateModel = async () => {
    setAccuracy(null);
    setConfusionMatrix(null);
    setEvaluationDone(false);
    try {
      const res = await fetch(DATA_URL);
      const data = await res.json();
      const labels = Array.from(new Set(data.map((d) => d.label))).sort();
      setAllLabels(labels);
      
      const shuffled = [...data].sort(() => 0.5 - Math.random());
      const trainCount = Math.floor(shuffled.length * 0.8);
      const trainSet = shuffled.slice(0, trainCount);
      const testSet = shuffled.slice(trainCount);
      const knn = new kNear(3);
      trainSet.forEach((ex) => knn.learn(ex.data, ex.label));
      
      let correct = 0;
      const matrix = {};
      labels.forEach((real) => (matrix[real] = {}));
      labels.forEach((real) => labels.forEach((predicted) => (matrix[real][predicted] = 0)));
      testSet.forEach((sample) => {
        const pred = knn.classify(sample.data);
        if (pred === sample.label) correct += 1;
        matrix[sample.label][pred] += 1;
      });
      setAccuracy(((correct / testSet.length) * 100).toFixed(2));
      setNumCorrect(correct);
      setNumTest(testSet.length);
      setConfusionMatrix(matrix);
      setEvaluationDone(true);
    } catch (e) {
      setError("Evaluation failed: " + e.message);
    }
  };

  // matrix table
  function ConfusionMatrixTable({ matrix, labels }) {
    return (
      <table style={{ borderCollapse: "collapse", marginTop: 16 }}>
        <thead>
          <tr>
            <th></th>
            {labels.map((real) => (
              <th key={real} style={{ border: "1px solid #999", padding: 4 }}>{real}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {labels.map((row) => (
            <tr key={row}>
              <th style={{ border: "1px solid #999", padding: 4 }}>{row}</th>
              {labels.map((col) => (
                <td key={col} style={{ border: "1px solid #999", padding: 4, textAlign: "center" }}>
                  {matrix[row][col]}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    );
  }

  useEffect(() => {
    return () => {
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
      if (videoRef.current && videoRef.current.srcObject) {
        videoRef.current.srcObject.getTracks().forEach((t) => t.stop());
      }
    };
  }, []);

  // UI
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
      <h2>Pose Classifier</h2>
      {error && <div style={{ color: "red", marginBottom: 8 }}>{error}</div>}
      {/* buttons */}
      <div style={{ display: "flex", gap: 16, marginBottom: 24 }}>
        <button
          onClick={trainModel}
          disabled={loading || modelTrained}
          style={{ padding: "10px 24px", fontWeight: "bold" }}
        >
          {modelTrained ? "Model trained" : loading ? "Training" : "Train model"}
        </button>
        <button
          onClick={handleStartWebcamAndPredict}
          disabled={!modelTrained || predictionLoading}
          style={{ padding: "10px 24px", fontWeight: "bold" }}
        >
          {predictionLoading ? "Starting" : "Start webcam & predict"}
        </button>
        <button
          onClick={evaluateModel}
          disabled={loading}
          style={{ padding: "10px 24px", fontWeight: "bold" }}
        >
          Evaluate model
        </button>
      </div>
      {/* webcam + canvas */}
      <div style={{
        display: "flex",
        flexDirection: "row",
        gap: 32,
        alignItems: "flex-start",
        justifyContent: "center",
        margin: "32px 0"
      }}>
        <div style={{ position: "relative", width: 480, height: 360 }}>
          <video
            ref={videoRef}
            style={{
              width: 480,
              height: 360,
              position: "absolute",
              top: 0,
              left: 0,
              background: "#000",
              transform: "scaleX(-1)",
            }}
            autoPlay
            muted
            playsInline
          />
          <canvas
            ref={canvasRef}
            style={{
              width: 480,
              height: 360,
              position: "absolute",
              top: 0,
              left: 0,
              pointerEvents: "none",
              transform: "scaleX(-1)",
            }}
          />
        </div>
        {/* matrix & accuracy */}
        {evaluationDone && (
          <div>
            <div style={{ fontSize: 20, fontWeight: "bold", marginBottom: 8 }}>
              Accuracy: {accuracy}% ({numCorrect} / {numTest} correct)
            </div>

            <div style={{ marginBottom: 8, fontSize: 18, fontWeight: "bold" }}>
              Confusion matrix:
            </div>
            {confusionMatrix && (
              <ConfusionMatrixTable matrix={confusionMatrix} labels={allLabels} />
            )}
          </div>
        )}
      </div>
      {/* prediction */}
      <div style={{ marginTop: 24, fontSize: 32, fontWeight: "bold", color: "#222" }}>
        {predictedStep ? `Predicted step: ${predictedStep}` : ""}
      </div>
    </div>
  );
}
