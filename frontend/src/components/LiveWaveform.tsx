import { useEffect, useRef } from 'react';
import type { FC } from 'react';

interface LiveWaveformProps {
  analyserNode: AnalyserNode | null;
  isActive: boolean;
  isPaused: boolean;
  height?: number;
}

export const LiveWaveform: FC<LiveWaveformProps> = ({
  analyserNode,
  isActive,
  isPaused,
  height = 70,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const animFrameRef = useRef<number | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let bufferLength = 128;
    let dataArray = new Uint8Array(bufferLength);

    if (analyserNode) {
      analyserNode.fftSize = 256;
      bufferLength = analyserNode.frequencyBinCount;
      dataArray = new Uint8Array(bufferLength);
    }

    const draw = () => {
      const width = canvas.width;
      const h = canvas.height;

      ctx.clearRect(0, 0, width, h);

      if (analyserNode && isActive && !isPaused) {
        analyserNode.getByteTimeDomainData(dataArray);

        // Neon glowing wave
        ctx.lineWidth = 2.5;
        const gradient = ctx.createLinearGradient(0, 0, width, 0);
        gradient.addColorStop(0, '#6366f1');
        gradient.addColorStop(0.5, '#a855f7');
        gradient.addColorStop(1, '#06b6d4');

        ctx.strokeStyle = gradient;
        ctx.shadowBlur = 8;
        ctx.shadowColor = 'rgba(168, 85, 247, 0.7)';

        ctx.beginPath();
        const sliceWidth = width / bufferLength;
        let x = 0;

        for (let i = 0; i < bufferLength; i++) {
          const v = dataArray[i] / 128.0; // 0 to 2
          const y = (v * h) / 2;

          if (i === 0) {
            ctx.moveTo(x, y);
          } else {
            ctx.lineTo(x, y);
          }

          x += sliceWidth;
        }

        ctx.stroke();
      } else {
        // Subtle resting flat line
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = 'rgba(148, 163, 184, 0.25)';
        ctx.shadowBlur = 0;

        ctx.beginPath();
        ctx.moveTo(0, h / 2);
        ctx.lineTo(width, h / 2);
        ctx.stroke();
      }

      animFrameRef.current = requestAnimationFrame(draw);
    };

    draw();

    return () => {
      if (animFrameRef.current) {
        cancelAnimationFrame(animFrameRef.current);
      }
    };
  }, [analyserNode, isActive, isPaused]);

  return (
    <div className="live-waveform-container" style={{ height: `${height}px` }}>
      <canvas
        ref={canvasRef}
        className="live-waveform-canvas"
        width={360}
        height={height}
      />
    </div>
  );
};
