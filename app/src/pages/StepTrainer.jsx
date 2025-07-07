import React, { useRef, useEffect, useState, useCallback } from "react";
import { useNavigate } from 'react-router-dom';
import { PoseLandmarker, FilesetResolver, DrawingUtils } from "@mediapipe/tasks-vision";
import kNear from "../lib/kNear";

const singleStep = [
  { name: "Step 1", gif: "/gif/jerkstep1.gif", label: "step_1" },
  { name: "Step 2", gif: "/gif/jerkstep2.gif", label: "step_2" },
  { name: "Step 3", gif: "/gif/jerkstep3.gif", label: "step_3" },
  { name: "Step 4", gif: "/gif/jerkstep4.gif", label: "step_4" },
];

const steps = Array(11).fill(singleStep).flat();
const LEG_INDICES = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32];
const MODEL_PATH = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task";
const WASM_PATH = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm";
const DATA_URL = "/data/jerk_pose_data.json";

// StepTrainer component
export default function StepTrainer() {
  const [stepIndex, setStepIndex] = useState(0);
  const stepIndexRef = useRef(0);
  const [timer, setTimer] = useState(3);
  const [isReady, setIsReady] = useState(false);
  const [poseCorrect, setPoseCorrect] = useState(false);
  const [showFinalStepButton, setShowFinalStepButton] = useState(false);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const poseLandmarkerRef = useRef(null);
  const knnRef = useRef(null);
  const frameRequestRef = useRef(null);
  const streamRef = useRef(null);
  const navigate = useNavigate();

  function getStepConfig(idx) {
    if (idx < 4) return { hold: 1.5, phase: 0 };
    if (idx < 8) return { hold: 1.5, phase: 1 };
    if (idx < 12) return { hold: 1.5, phase: 2 };
    if (idx < 16) return { hold: 1, phase: 3 };
    if (idx < 20) return { hold: 1, phase: 4 };
    if (idx < 24) return { hold: 1, phase: 5 };
    if (idx < 28) return { hold: 0.5, phase: 6 };
    if (idx < 32) return { hold: 0.5, phase: 7 };
    if (idx < 36) return { hold: 0.2, phase: 8 };
    if (idx < 40) return { hold: 0.2, phase: 9 };
    return { hold: 0.2, phase: 10 };
  }

  const phaseTexts = [
    { title: "Face the right way and imitate the movement you see" },
    { title: "Well done! Try stay in the middle" },
    { title: "Memorize the steps" },
    { title: "Let's go a bit faster now" },
    { title: "Good job, keep going!" },
    { title: "try to do it  without looking at the example" },
    { title: "Faster!" },
    { title: "FASTAAAAA!!" },
    { title: "Let the steps flow into eachother!" },
    { title: "This is speed you'll have to do it!" },
    { title: "Almost got it!!" },
  ];

  useEffect(() => {
    stepIndexRef.current = stepIndex;
  }, [stepIndex]);

  // setup mediapipe pose landmarker and webcam stream
  useEffect(() => {
    let stopped = false;
    async function setup() {
      const vision = await FilesetResolver.forVisionTasks(WASM_PATH);
      poseLandmarkerRef.current = await PoseLandmarker.createFromOptions(vision,
        {
          baseOptions: { modelAssetPath: MODEL_PATH, delegate: "GPU" },
          runningMode: "VIDEO",
          numPoses: 1,
        }
      );
      // load knn for classicication & trained data
      const kNear = (await import("../lib/kNear")).default;
      const data = await (await fetch(DATA_URL)).json();
      const knn = new kNear(1);
      data.forEach((item) => knn.learn(item.data, item.label)); // learn = load in
      knnRef.current = knn;

      // start webcam
      if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await new Promise((resolve) => {
            videoRef.current.onloadedmetadata = () => resolve();
          });
          await videoRef.current.play().catch(() => { });
        }
      }
      setIsReady(true);
      if (!stopped) {
        frameRequestRef.current = requestAnimationFrame(predictLoop);
      }
    }
    setup();
    return () => {
      stopped = true;
      if (frameRequestRef.current) cancelAnimationFrame(frameRequestRef.current);
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  // prediction loop
  const predictLoop = useCallback(async () => {
    if (
      !poseLandmarkerRef.current || !videoRef.current || !canvasRef.current || !knnRef.current || stepIndexRef.current >= steps.length
    ) {
      return;
    }
    const curStepIdx = stepIndexRef.current;
    const video = videoRef.current;
    if (video.readyState === 4) {
      const width = video.videoWidth;
      const height = video.videoHeight;
      canvasRef.current.width = width;
      canvasRef.current.height = height;
      const now = performance.now();
      const result = await poseLandmarkerRef.current.detectForVideo(video, now);
      let detectedLegData = [];
      let poseMatch = false;
      let targetLabel = steps[curStepIdx].label;

      if (result && result.landmarks && result.landmarks[0]) {
        detectedLegData = LEG_INDICES.flatMap((idx) => [
          result.landmarks[0][idx]?.x ?? 0,
          result.landmarks[0][idx]?.y ?? 0,
        ]);
        const predictedLabel = knnRef.current.classify(detectedLegData);
        poseMatch = String(predictedLabel).trim() === String(targetLabel).trim();
      }
      setPoseCorrect(poseMatch);
      drawOnCanvas(result, width, height, poseMatch);
    }
    frameRequestRef.current = requestAnimationFrame(predictLoop);
  }, []);

  // loop through steps
  useEffect(() => {
    if (stepIndex >= steps.length) {
      setShowFinalStepButton(true);
      return;
    }
    if (!poseCorrect) return;

    if (timer <= 0) {
      if (stepIndex < steps.length - 1) {
        setStepIndex((idx) => idx + 1);
        setTimer(getStepConfig(stepIndex + 1).hold);
        setPoseCorrect(false);
      } else if (stepIndex === steps.length - 1) {
        setStepIndex((idx) => idx + 1);
      }
      return;
    }

    const timeout = setTimeout(() => setTimer((t) => t - 0.1), 100);
    return () => clearTimeout(timeout);
  }, [poseCorrect, timer, stepIndex]);

  useEffect(() => {
    if (stepIndex >= steps.length) return;
    setTimer(getStepConfig(stepIndex).hold);
  }, [stepIndex]);

  // draw on canvas
  function drawOnCanvas(result, width, height, poseMatch) {
    const ctx = canvasRef.current.getContext("2d");
    ctx.save();
    ctx.clearRect(0, 0, width, height);

    ctx.beginPath();
    ctx.moveTo(width / 2, 0);
    ctx.lineTo(width / 2, height);
    ctx.strokeStyle = "#2479f5";
    ctx.lineWidth = 4;
    ctx.stroke();

    if (result && result.landmarks && result.landmarks[0]) {
      ctx.save();
      ctx.strokeStyle = poseMatch ? "#26c165" : "#e53e3e";
      ctx.fillStyle = poseMatch ? "#26c165" : "#e53e3e";
      ctx.lineWidth = 6;
      LEG_INDICES.forEach((idx) => {
        const pt = result.landmarks[0][idx];
        ctx.beginPath();
        ctx.arc(pt.x * width, pt.y * height, 8, 0, 2 * Math.PI);
        ctx.fill();
      });
      for (let i = 1; i < LEG_INDICES.length; i++) {
        const pt1 = result.landmarks[0][LEG_INDICES[i - 1]];
        const pt2 = result.landmarks[0][LEG_INDICES[i]];
        ctx.beginPath();
        ctx.moveTo(pt1.x * width, pt1.y * height);
        ctx.lineTo(pt2.x * width, pt2.y * height);
        ctx.stroke();
      }
      ctx.restore();

      const drawingUtils = new DrawingUtils(ctx);
      drawingUtils.drawLandmarks(result.landmarks[0], {
        color: "#aaa",
        radius: 4,
      });
      drawingUtils.drawConnectors(
        result.landmarks[0],
        PoseLandmarker.POSE_CONNECTIONS,
        { color: "#ccc", lineWidth: 2 }
      );
    }
    ctx.restore();
  }

  const currentStep = steps[stepIndex];
  const { phase } = getStepConfig(stepIndex);

  // UI
  return (
    <div
    
      style={{
        minHeight: "100vh",
        minWidth: "100vw",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        position: "relative",
      }}
    >
      {/* intructions */}
      <div style={{ marginBottom: 20, textAlign: "center" }}>
        <h2 style={{ fontWeight: 700, fontSize: 48, background: "rgba(255, 255, 255, 0.3)", padding: "2px 2px", borderRadius: 12, boxShadow: "0 2px 8px #0001", color: "#223" }}>
          {phaseTexts[phase].title}
        </h2>
      </div>
      <div
        style={{
          display: "flex",
          flexDirection: "row",
          alignItems: "flex-start",
          justifyContent: "center",
          width: "100%",
          maxWidth: 1400,
        }}
      >

        {/* webcam + overlay */}
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            marginRight: 30,
          }}
        >
          <div
            style={{
              position: "relative",
              width: 640,
              height: 480,
              borderRadius: 18,
              boxShadow: "0 4px 24px #0002",
              background: "#222",
              overflow: "hidden",
            }}
          >

            {/* timer */}
            <div
              style={{
                position: "absolute",
                top: 26,
                left: "50%",
                transform: "translateX(-50%)",
                fontSize: 58,
                fontWeight: 900,
                color: poseCorrect ? "#26c165" : "#e53e3e", // correct pose check
                background: "rgba(255, 255, 255, 0.5)",
                borderRadius: 24,
                padding: "8px 40px",
                zIndex: 10,
                boxShadow: "0 2px 24px #0002",
                minWidth: 100,
                textAlign: "center",
                pointerEvents: "none",
                userSelect: "none",
              }}
            >
              {timer > 0 ? timer.toFixed(1) : "0"}
            </div>
            {/* webcam video / live feed element */}
            <video
              ref={videoRef}
              style={{
                width: 640,
                height: 480,
                position: "absolute",
                top: 0,
                left: 0,
                background: "#000",
                transform: "scaleX(-1)", // mirror
                borderRadius: 18,
                zIndex: 1,
              }}
              autoPlay
              muted
              playsInline
            />
            {/* pose overlay */}
            <canvas
              ref={canvasRef}
              style={{
                width: 640,
                height: 480,
                position: "absolute",
                top: 0,
                left: 0,
                pointerEvents: "none",
                transform: "scaleX(-1)", //mirror
                borderRadius: 18,
                zIndex: 2,
              }}
            />
            {/* Loading */}
            {!isReady && (
              <span style={{ color: "#fff", fontSize: 18, zIndex: 3, position: "absolute", top: 18, left: 20 }}>
                loading camera + model...
              </span>
            )}
          </div>
        </div>

        {/* step gif + label */}
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            minWidth: 220,
            height: 480,
            justifyContent: "flex-start",
            position: "relative",
          }}
        >
          <div
            style={{
              fontSize: 28,
              fontWeight: 700,
              color: "#223",
              zIndex: 5,
              background: "rgba(255, 255, 255, 0.5)",
              borderRadius: 12,
              padding: "8px 32px",
              marginBottom: 14,
              marginTop: 4,
              pointerEvents: "none",
              boxShadow: "0 2px 8px #0001",
              textAlign: "center",
              minWidth: 140,
              position: "relative",
            }}
          >
            {currentStep?.name}
          </div>
          <img
            src={currentStep?.gif}
            alt={currentStep?.name}
            style={{
              height: 480,
              width: "auto",
              objectFit: "cover",
              borderRadius: 16,
              boxShadow: "0 2px 16px #0001",
              background: "#eee",
              display: "block",
              marginTop: 0,
            }}
          />
        </div>
      </div>
      {/* final step button */}
      {showFinalStepButton && (
        <button
          onClick={() => navigate('/final')}
          style={{
            position: "fixed",
            top: "50%",
            left: "50%",
            transform: "translate(-50%,-50%)",
            fontSize: 36,
            padding: "20px 60px",
            background: "#2479f5",
            color: "#fff",
            border: "none",
            borderRadius: 16,
            fontWeight: 700,
            boxShadow: "0 4px 24px #0003",
            zIndex: 1000,
            cursor: "pointer",
          }}
        >
          Now for real!
        </button>
      )}
    </div>
  );
}
