import React, { useEffect } from "react";
import { useNavigate } from "react-router-dom";

export default function Home() {
    const navigate = useNavigate();

    // embed tenor gif
    useEffect(() => {
        if (!window.Tenor) {
            const script = document.createElement("script");
            script.src = "https://tenor.com/embed.js";
            script.async = true;
            document.body.appendChild(script);
        } else if (window.Tenor?.EmbedHandler) {
            window.Tenor.EmbedHandler.process();
        }
    }, []);

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
            }}
        >
            <h1
                style={{
                    fontSize: 40,
                    fontWeight: 800,
                    marginBottom: 32,
                    marginTop: 0,
                    textAlign: "center",
                    color: "#243142",
                }}
            >
                WYM you can't jerk in 2025?!?!?!?!?
            </h1>
            {/* tenor gif */}
            <div
                style={{
                    marginBottom: "40px",
                    borderRadius: "24px",
                    boxShadow: "0 2px 16px #0002",
                    overflow: "hidden",
                    width: "420px",
                    maxWidth: "90vw",
                }}
                dangerouslySetInnerHTML={{
                    __html: `
                        <div class="tenor-gif-embed"
                             data-postid="14852991766593389876"
                             data-share-method="host"
                             data-aspect-ratio="1.1422"
                             data-width="100%">
                        </div>
                    `,
                }}
            />
            {/* start button */}
            <button
                style={{
                    padding: "16px 40px",
                    fontSize: 22,
                    background: "#2479f5",
                    color: "#fff",
                    border: "none",
                    borderRadius: 12,
                    fontWeight: "bold",
                    cursor: "pointer",
                    boxShadow: "0 2px 12px #0001",
                    transition: "background 0.2s",
                }}
                onClick={() => navigate("/train")}
            >
                Teach me how to jerk
            </button>
        </div>
    );
}
