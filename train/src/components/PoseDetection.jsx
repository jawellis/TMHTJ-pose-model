import React, { useRef, useEffect, useState } from "react";
import { PoseLandmarker, FilesetResolver, DrawingUtils } from "@mediapipe/tasks-vision";

const STEPS = ["step_1", "step_2", "step_3", "step_4"];
const LEG_INDICES = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32];
const MODEL_PATH = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task";
const WASM_PATH = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm";

// PoseDetection component 
export default function PoseDetection() {
  const camRef = useRef(null);
  const canvas = useRef(null);
  const poseLandmarkerRef = useRef(null);
  const [camOn, setCamOn] = useState(false);
  const [loading, setLoading] = useState(true);
  const [activeStep, setActiveStep] = useState("");
  const [timer, setTimer] = useState(0);
  const [isRecording, setIsRecording] = useState(false);
  const [dataSet, setDataSet] = useState([]);
  const [recordingDone, setRecordingDone] = useState("");

  // model setup 
  useEffect(() => {
    async function initModel() {
      setLoading(true);
      const vision = await FilesetResolver.forVisionTasks(WASM_PATH);
      poseLandmarkerRef.current = await PoseLandmarker.createFromOptions(vision,
        {
          baseOptions: { modelAssetPath: MODEL_PATH, delegate: "GPU" },
          runningMode: "VIDEO",
          numPoses: 1
        });
      setLoading(false);
    }
    initModel();
    return () => {
      if (camOn) handleCamToggle();
    };
  }, []);

  // start detection if cam on
  useEffect(() => {
    if (camOn) detectPose();
  }, [camOn, isRecording, activeStep]);

  // webcam on / off
  const handleCamToggle = async () => {
    if (!camOn) {
      // on
      if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (camRef.current) {
          camRef.current.srcObject = stream;
          await camRef.current.play();
        }
      }
      setCamOn(true);
    } else {
      // off
      if (camRef.current && camRef.current.srcObject) {
        camRef.current.srcObject.getTracks().forEach(track => track.stop());
        camRef.current.srcObject = null;
      }
      setCamOn(false);
      clearCanvas();
    }
  };

  // 5s recording after 5s countdown
  const startRecording = (step) => {
    setActiveStep(step);
    setRecordingDone("");
    setTimer(5);
    setIsRecording(false);
    setDataSet([]);

    const countdown = setInterval(() => {
      setTimer(prev => {
        if (prev <= 1) {
          clearInterval(countdown);
          setIsRecording(true);
          setTimeout(() => {
            setIsRecording(false);
            setActiveStep("");
            setRecordingDone(step);
          }, 5000);
          return 0;
        }
        return prev - 1;
      });
    }, 1000);
  };

  // detect poses
  const detectPose = async () => {
    if (!camOn || !poseLandmarkerRef.current) return;
    if (camRef.current.readyState === 4) {
      const video = camRef.current;
      const width = video.videoWidth;
      const height = video.videoHeight;
      canvas.current.width = width;
      canvas.current.height = height;
      const now = performance.now();
      const result = await poseLandmarkerRef.current.detectForVideo(video, now);
      drawOnCanvas(result, width, height);

      if (
        isRecording &&
        activeStep &&
        result.landmarks &&
        result.landmarks[0]
      ) {
        const landmarks = result.landmarks[0];
        const data = LEG_INDICES.flatMap(idx => [landmarks[idx].x, landmarks[idx].y]); // extract leg landmarks
        setDataSet(ds => [...ds, { label: activeStep, data }]);
      }
    }
    if (camOn) requestAnimationFrame(detectPose);
  };
  // draw on canvas
  const drawOnCanvas = (result, width, height) => {
    const ctx = canvas.current.getContext("2d");
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
      result.landmarks.forEach(landmarkSet => {
        drawingUtils.drawLandmarks(landmarkSet, { radius: 6 });
        drawingUtils.drawConnectors(landmarkSet, PoseLandmarker.POSE_CONNECTIONS);
      });
    }
    ctx.restore();
  };

  //  save dataset to JSON
  const handleSave = () => {
    if (!dataSet.length) return;
    const filename = `jerk_data_${dataSet[0].label}.json`;
    const blob = new Blob([JSON.stringify(dataSet, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const clearCanvas = () => {
    const ctx = canvas.current?.getContext("2d");
    if (ctx) ctx.clearRect(0, 0, canvas.current.width, canvas.current.height);
  };

  // UI
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
      <h2>Pose Detection</h2>
      {loading ? (
        <p>Loading model…</p>
      ) : (
        <>
          <div style={{ position: "relative", width: 480, height: 360 }}>
            <video
              ref={camRef}
              style={{
                width: 480,
                height: 360,
                position: "absolute",
                top: 0,
                left: 0,
                background: "#000",
                transform: "scaleX(-1)"
              }}
              autoPlay
              muted
              playsInline
            />
            <canvas
              ref={canvas}
              style={{
                width: 480,
                height: 360,
                position: "absolute",
                top: 0,
                left: 0,
                pointerEvents: "none",
                transform: "scaleX(-1)"
              }}
            />
          </div>
          <button onClick={handleCamToggle} style={{ marginTop: 16 }}>
            {camOn ? "Stop camera" : "Start camera"}
          </button>

          {/* button color change for recording */}
          <div style={{ margin: "32px 0" }}>
            {STEPS.map((step) => (
              <button
                key={step}
                onClick={() => startRecording(step)}
                disabled={isRecording || timer > 0 || !camOn}
                style={{
                  margin: "0 8px",
                  padding: "8px 16px",
                  background: recordingDone === step ? "#8df5a4" : "#e1e1e1",
                  color: "#222",
                  border: "1px solid #888",
                  borderRadius: 8,
                  fontWeight: "bold",
                  opacity: isRecording || timer > 0 || !camOn ? 0.6 : 1,
                  cursor: isRecording || timer > 0 || !camOn ? "not-allowed" : "pointer"
                }}
              >
                {`Record ${step.replace("_", " ")}`}
              </button>
            ))}
          </div>

          {/* countdown */}
          {timer > 0 && (
            <div style={{ fontSize: 24, fontWeight: 700, color: "#ff7200" }}>
              Recording in {timer}
            </div>
          )}
          {/* recording */}
          {isRecording && (
            <div style={{ fontSize: 24, fontWeight: 700, color: "#0080ff" }}>
              Recording {activeStep}
            </div>
          )}
          <button
            onClick={handleSave}
            disabled={!dataSet.length}
            style={{
              marginTop: 24,
              background: "#fff",
              color: "#000",
              border: "2px solid #888",
              borderRadius: 8,
              padding: "8px 20px",
              fontWeight: "bold",
              fontSize: 18,
              cursor: !dataSet.length ? "not-allowed" : "pointer"
            }}
          >
            Save samples
          </button>
        </>
      )}
    </div>
  );
}
