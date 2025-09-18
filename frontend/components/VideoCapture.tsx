// frontend\components\VideoCapture.tsx
import React, { useRef, useState } from "react";
import { verifyFace } from "../app/api/connection-details/faceApi";
import FaceVerifyResult from "./FaceVerifyResult";

const VideoCapture: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [result, setResult] = useState<any>(null);

  // Start webcam
  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch (err) {
      console.error("Camera access denied:", err);
    }
  };

  // Capture snapshot and send to backend
  const captureAndVerify = async () => {
    if (!canvasRef.current || !videoRef.current) return;

    const ctx = canvasRef.current.getContext("2d");
    if (!ctx) return;

    // Draw video frame onto canvas
    ctx.drawImage(videoRef.current, 0, 0, canvasRef.current.width, canvasRef.current.height);

    // Convert to Blob
    canvasRef.current.toBlob(async (blob) => {
      if (!blob) return;
      const response = await verifyFace(blob);
      setResult(response);
    }, "image/jpeg");
  };

  return (
    <div className="p-4">
      <h2 className="text-xl font-bold mb-2">Face Recognition</h2>

      <video ref={videoRef} autoPlay className="border rounded w-64 h-48" />
      <canvas ref={canvasRef} width={320} height={240} className="hidden" />

      <div className="mt-2 flex gap-2">
        <button
          onClick={startCamera}
          className="px-4 py-2 bg-blue-500 text-white rounded"
        >
          Start Camera
        </button>

        <button
          onClick={captureAndVerify}
          className="px-4 py-2 bg-green-500 text-white rounded"
        >
          Verify Face
        </button>
      </div>

      {result && <FaceVerifyResult result={result} />}
    </div>
  );
};

export default VideoCapture;
