import React, { useRef, useEffect, useState, useCallback } from "react";
import { PoseLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";

const steps = [
    { name: "Step 1", gif: "/gif/jerkstep1.gif", label: "step_1" },
    { name: "Step 2", gif: "/gif/jerkstep2.gif", label: "step_2" },
    { name: "Step 3", gif: "/gif/jerkstep3.gif", label: "step_3" },
    { name: "Step 4", gif: "/gif/jerkstep4.gif", label: "step_4" },
];

const LEG_INDICES = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32];
const MODEL_PATH = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task";
const WASM_PATH = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm";
const DATA_URL = "/data/jerk_pose_data.json";
const VIDEO_PATH = "/video/jerk-music-video.mp4";

//final step component
export default function FinalStep() {
    const [stepIndex, setStepIndex] = useState(0);
    const [poseCorrect, setPoseCorrect] = useState(false);
    const [isReady, setIsReady] = useState(false);
    const [timer, setTimer] = useState(0.2);
    const [countdown, setCountdown] = useState(10);
    const [videoEnded, setVideoEnded] = useState(false);
    const [showIntro, setShowIntro] = useState(true);
    const [videoStarted, setVideoStarted] = useState(false);
    const videoRef = useRef(null);
    const stepIndexRef = useRef(0);
    const frameRequestRef = useRef(null);
    const countdownRef = useRef();
    const canvasRef = useRef(null);
    const videoWebcamRef = useRef(null);
    const poseLandmarkerRef = useRef(null);
    const knnRef = useRef(null);
    const streamRef = useRef(null);

    useEffect(() => {
        stepIndexRef.current = stepIndex;
    }, [stepIndex]);

    // setup model and webcam
    useEffect(() => {
        let stopped = false;
        async function setup() {
            const vision = await FilesetResolver.forVisionTasks(WASM_PATH);
            poseLandmarkerRef.current = await PoseLandmarker.createFromOptions(
                vision,
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
            data.forEach((item) => knn.learn(item.data, item.label));
            knnRef.current = knn;

            if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
                const stream = await navigator.mediaDevices.getUserMedia({ video: true });
                streamRef.current = stream;
                if (videoWebcamRef.current) {
                    videoWebcamRef.current.srcObject = stream;
                    await new Promise((resolve) => {
                        videoWebcamRef.current.onloadedmetadata = () => resolve();
                    });
                    await videoWebcamRef.current.play().catch(() => { });
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
            !poseLandmarkerRef.current || !videoWebcamRef.current || !canvasRef.current || !knnRef.current
        )
            return;
        const curStepIdx = stepIndexRef.current;
        const video = videoWebcamRef.current;
        if (video.readyState === 4) {
            const width = video.videoWidth;
            const height = video.videoHeight;
            canvasRef.current.width = width;
            canvasRef.current.height = height;
            // data current frame
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
                const predicted = knnRef.current.classify(detectedLegData);
                poseMatch = String(predicted).trim() === String(targetLabel).trim();
            }
            setPoseCorrect(poseMatch);
            drawOnCanvas(result, width, height, poseMatch);
        }
        frameRequestRef.current = requestAnimationFrame(predictLoop);
    }, []);

    // loop through steps
    useEffect(() => {
        if (showIntro || countdown > 0 || videoEnded) return;
        if (!poseCorrect) return;
        if (timer <= 0) {
            setStepIndex((idx) => (idx + 1) % steps.length);
            setTimer(0.2);
            setPoseCorrect(false);
            return;
        }
        const timeout = setTimeout(() => setTimer((t) => t - 0.2), 100);
        return () => clearTimeout(timeout);
    }, [poseCorrect, timer, stepIndex, countdown, videoEnded, showIntro]);

    useEffect(() => {
        if (!videoStarted) return;
        setCountdown(10);
        setVideoEnded(false);
        if (videoRef.current) {
            videoRef.current.currentTime = 0;
            videoRef.current.muted = false;
            videoRef.current.play().catch(() => { });
        }
        countdownRef.current = setInterval(() => {
            setCountdown((t) => {
                if (t <= 1) {
                    clearInterval(countdownRef.current);
                    return 0;
                }
                return t - 1;
            });
        }, 1000);
        return () => clearInterval(countdownRef.current);
    }, [videoStarted]);

    const handleVideoEnded = () => setVideoEnded(true);

    // reset for try again
    const handleTryAgain = () => {
        setCountdown(10);
        setStepIndex(0);
        setTimer(0.2);
        setPoseCorrect(false);
        setVideoEnded(false);
        setVideoStarted(true);
        setShowIntro(false);
        if (videoRef.current) {
            videoRef.current.currentTime = 0;
            videoRef.current.muted = false;
            videoRef.current.play().catch(() => { });
        }
        countdownRef.current = setInterval(() => {
            setCountdown((t) => {
                if (t <= 1) {
                    clearInterval(countdownRef.current);
                    return 0;
                }
                return t - 1;
            });
        }, 1000);
    };

    // draw on canvas
    function drawOnCanvas(result, width, height, poseMatch) {
        if (!canvasRef.current) return;
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
        }
        ctx.restore();
    }

    // UI
    return (
        <div
            style={{
                minHeight: "100vh",
                minWidth: "100vw",
                position: "relative",
                overflow: "hidden",
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "center",
                background: "#000",
            }}
        >
            {/* background video clip */}
            <video
                ref={videoRef}
                src={VIDEO_PATH}
                style={{
                    position: "fixed",
                    top: 0,
                    left: 0,
                    width: "100vw",
                    height: "100vh",
                    objectFit: "cover",
                    opacity: 0.33,
                    zIndex: 0,
                    pointerEvents: "none",
                }}
                playsInline
                onEnded={handleVideoEnded}
            />

            {/* intro overlay */}
            {showIntro && (
                <div
                    style={{
                        position: "fixed",
                        top: 0,
                        left: 0,
                        width: "100vw",
                        height: "100vh",
                        background: "rgba(25,25,25,0.86)",
                        zIndex: 10,
                        display: "flex",
                        flexDirection: "column",
                        alignItems: "center",
                        justifyContent: "center",
                    }}
                >
                    <h1
                        style={{
                            color: "#fff",
                            fontWeight: 800,
                            fontSize: 48,
                            marginBottom: 40,
                            textShadow: "0 4px 28px #000b",
                            textAlign: "center",
                        }}
                    >
                        Let's try it with music now.<br />
                        Are you ready?
                    </h1>
                    <button
                        style={{
                            fontSize: 28,
                            padding: "18px 54px",
                            background: "#2479f5",
                            color: "#fff",
                            border: "none",
                            borderRadius: 18,
                            fontWeight: "bold",
                            cursor: "pointer",
                            boxShadow: "0 2px 18px #0005",
                        }}
                        onClick={() => {
                            setShowIntro(false);
                            setVideoStarted(true);
                            if (videoRef.current) {
                                videoRef.current.currentTime = 0;
                                videoRef.current.muted = false;
                                videoRef.current.play().catch(() => { });
                            }
                        }}
                    >
                        Start
                    </button>
                </div>
            )}

            <div
                style={{
                    width: "100vw",
                    height: "100vh",
                    position: "relative",
                    zIndex: 1,
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                    justifyContent: "center",
                }}
            >
                {/* title */}
                <div
                    style={{
                        position: "absolute",
                        top: 60,
                        left: 0,
                        right: 0,
                        zIndex: 2,
                        textAlign: "center",
                        pointerEvents: "none",
                    }}
                >
                    <h1
                        style={{
                            fontSize: 52,
                            color: "#fff",
                            fontWeight: 800,
                            textShadow: "0 3px 18px #0009",
                            margin: 0,
                        }}
                    >
                        Jerrrrrkkkkkkk
                    </h1>
                </div>

                <div
                    style={{
                        display: "flex",
                        flexDirection: "row",
                        alignItems: "center",
                        justifyContent: "center",
                        width: "100%",
                        maxWidth: 1400,
                        zIndex: 2,
                        margin: "0 auto",
                        marginTop: 120,
                    }}
                >

                    <div
                        style={{
                            margin: "0 60px",
                            display: "flex",
                            flexDirection: "column",
                            alignItems: "center",
                        }}
                    >
                        {/* canvas / webcam */}
                        <div
                            style={{
                                width: 640,
                                height: 480,
                                background: "#222",
                                borderRadius: 18,
                                boxShadow: "0 4px 24px #0002",
                                marginBottom: 16,
                                display: "flex",
                                alignItems: "center",
                                justifyContent: "center",
                                position: "relative",
                            }}
                        >
                            <video
                                ref={videoWebcamRef}
                                style={{
                                    width: 640,
                                    height: 480,
                                    position: "absolute",
                                    top: 0,
                                    left: 0,
                                    background: "#000",
                                    transform: "scaleX(-1)",
                                    borderRadius: 18,
                                    zIndex: 1,
                                }}
                                autoPlay
                                muted
                                playsInline
                            />
                            <canvas
                                ref={canvasRef}
                                style={{
                                    width: 640,
                                    height: 480,
                                    position: "absolute",
                                    top: 0,
                                    left: 0,
                                    pointerEvents: "none",
                                    transform: "scaleX(-1)",
                                    borderRadius: 18,
                                    zIndex: 2,
                                }}
                            />
                            {!isReady && (
                                <span style={{ color: "#fff", fontSize: 18, zIndex: 3 }}>
                                    Loading camera + model...
                                </span>
                            )}
                        </div>
                        {/* step gifs*/}
                        <div
                            style={{
                                display: "flex",
                                flexDirection: "row",
                                gap: 18,
                                marginTop: 16,
                                justifyContent: "center",
                                alignItems: "end"
                            }}
                        >
                            {steps.map((step, idx) => (
                                <div
                                    key={idx}
                                    style={{
                                        display: "flex",
                                        flexDirection: "column",
                                        alignItems: "center",
                                    }}
                                >
                                    <img
                                        src={step.gif}
                                        alt={step.name}
                                        style={{
                                            height: 70,
                                            width: "auto",
                                            borderRadius: 10,
                                            boxShadow: "0 2px 8px #0008",
                                            background: "#eee",
                                            display: "block",
                                            border: stepIndex === idx ? "3px solid #2479f5" : "3px solid transparent",
                                            transition: "border 0.2s"
                                        }}
                                    />
                                    <div
                                        style={{
                                            fontSize: 16,
                                            color: "#fff",
                                            fontWeight: 600,
                                            marginTop: 4,
                                            textShadow: "0 2px 6px #000d",
                                        }}
                                    >
                                        {step.name}
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>

                {/* countdown overlay */}
                {countdown > 0 && (
                    <div
                        style={{
                            position: "fixed",
                            top: "50%",
                            left: "50%",
                            transform: "translate(-50%, -50%)",
                            zIndex: 9999,
                            pointerEvents: "none",
                            fontSize: 92,
                            fontWeight: 900,
                            color: "#fff",
                            textShadow: "0 5px 24px #000a",
                            background: "rgba(20,20,20,0.44)",
                            padding: "40px 120px",
                            borderRadius: 48,
                            minWidth: 120,
                            textAlign: "center",
                            userSelect: "none",
                        }}
                    >
                        {countdown}
                    </div>
                )}

                {/* try again overlay" */}
                {videoEnded && (
                    <div
                        style={{
                            position: "absolute",
                            top: "48%",
                            left: "50%",
                            transform: "translate(-50%, -50%)",
                            zIndex: 10,
                            background: "rgba(22,22,22,0.9)",
                            padding: 40,
                            borderRadius: 28,
                            boxShadow: "0 6px 32px #0009",
                            textAlign: "center",
                        }}
                    >
                        <h2
                            style={{
                                fontSize: 38,
                                color: "#fff",
                                fontWeight: 900,
                                margin: "0 0 16px 0",
                                textShadow: "0 2px 8px #000c",
                            }}
                        >
                            Try again?
                        </h2>
                        <button
                            style={{
                                fontSize: 26,
                                padding: "16px 46px",
                                background: "#2479f5",
                                color: "#fff",
                                border: "none",
                                borderRadius: 14,
                                fontWeight: "bold",
                                cursor: "pointer",
                                boxShadow: "0 2px 18px #0004",
                            }}
                            onClick={handleTryAgain}
                        >
                            Play Again
                        </button>
                    </div>
                )}
            </div>
        </div>
    );
}
