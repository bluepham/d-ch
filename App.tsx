import React, { useState, useRef, useCallback, ChangeEvent } from 'react';
import { GoogleGenAI, LiveSession, LiveServerMessage, Blob, Modality } from "@google/genai";
import { MicrophoneIcon, StopIcon, LoadingSpinnerIcon, WarningIcon, UploadCloudIcon, ClipboardIcon, FileMusicIcon } from './components/icons';

// Helper function to encode Uint8Array to base64
function encode(bytes: Uint8Array): string {
  let binary = '';
  const len = bytes.byteLength;
  for (let i = 0; i < len; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

// Helper function to create a Blob in the format required by the API
function createBlob(data: Float32Array): Blob {
  const l = data.length;
  const int16 = new Int16Array(l);
  for (let i = 0; i < l; i++) {
    int16[i] = data[i] * 32768;
  }
  return {
    data: encode(new Uint8Array(int16.buffer)),
    mimeType: 'audio/pcm;rate=16000',
  };
}


export default function App() {
  const [mode, setMode] = useState<'live' | 'file'>('live');

  // Live transcription states
  const [isListening, setIsListening] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [currentTranscript, setCurrentTranscript] = useState('');
  const [transcriptHistory, setTranscriptHistory] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);

  // File transcription states
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [fileTranscript, setFileTranscript] = useState('');
  const [fileProcessingState, setFileProcessingState] = useState<'idle' | 'processing' | 'success' | 'error'>('idle');
  const [fileError, setFileError] = useState<string | null>(null);
  const [isCopied, setIsCopied] = useState(false);

  const sessionPromiseRef = useRef<Promise<LiveSession> | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const scriptProcessorRef = useRef<ScriptProcessorNode | null>(null);
  const mediaStreamSourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);

  const stopListening = useCallback(async () => {
    if (!sessionPromiseRef.current) return;

    try {
        const session = await sessionPromiseRef.current;
        session.close();
    } catch (e) {
        console.error("Error closing session:", e);
    } finally {
        if (mediaStreamRef.current) {
            mediaStreamRef.current.getTracks().forEach(track => track.stop());
            mediaStreamRef.current = null;
        }
        if (scriptProcessorRef.current) {
            scriptProcessorRef.current.disconnect();
            scriptProcessorRef.current = null;
        }
        if (mediaStreamSourceRef.current) {
            mediaStreamSourceRef.current.disconnect();
            mediaStreamSourceRef.current = null;
        }
        if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
            await audioContextRef.current.close();
            audioContextRef.current = null;
        }
        
        sessionPromiseRef.current = null;
        setIsListening(false);
        setIsConnecting(false);
        
        if (currentTranscript.trim()) {
            setTranscriptHistory(prev => [...prev, currentTranscript.trim()]);
        }
        setCurrentTranscript('');
    }
  }, [currentTranscript]);


  const startListening = useCallback(async () => {
    if (isListening || isConnecting) return;
    
    setError(null);
    setIsConnecting(true);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaStreamRef.current = stream;

      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY as string });
      
      const sessionPromise = ai.live.connect({
        model: 'gemini-2.5-flash-native-audio-preview-09-2025',
        callbacks: {
          onopen: () => {
              setIsConnecting(false);
              setIsListening(true);
              
              audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });
              mediaStreamSourceRef.current = audioContextRef.current.createMediaStreamSource(stream);
              scriptProcessorRef.current = audioContextRef.current.createScriptProcessor(4096, 1, 1);
              
              scriptProcessorRef.current.onaudioprocess = (audioProcessingEvent) => {
                const inputData = audioProcessingEvent.inputBuffer.getChannelData(0);
                const pcmBlob = createBlob(inputData);
                
                if (sessionPromiseRef.current) {
                    sessionPromiseRef.current.then((session) => {
                        session.sendRealtimeInput({ media: pcmBlob });
                    }).catch(e => {
                        console.error("Error sending audio data:", e);
                        setError("Failed to send audio data.");
                        stopListening();
                    });
                }
              };
              
              mediaStreamSourceRef.current.connect(scriptProcessorRef.current);
              scriptProcessorRef.current.connect(audioContextRef.current.destination);
          },
          onmessage: (message: LiveServerMessage) => {
              if (message.serverContent?.inputTranscription) {
                  const text = message.serverContent.inputTranscription.text;
                  setCurrentTranscript(prev => prev + text);
              }

              if (message.serverContent?.turnComplete) {
                if(currentTranscript.trim()) {
                  setTranscriptHistory(prev => [...prev, currentTranscript.trim()]);
                }
                setCurrentTranscript('');
              }
          },
          onerror: (e: ErrorEvent) => {
            console.error("Session error:", e);
            setError(`Connection error: ${e.message}. Please try again.`);
            stopListening();
          },
          onclose: (e: CloseEvent) => {
            stopListening();
          },
        },
        config: {
          inputAudioTranscription: {},
          responseModalities: [Modality.AUDIO],
        },
      });

      sessionPromiseRef.current = sessionPromise;

    } catch (err) {
      console.error("Failed to start listening:", err);
      let message = 'An unknown error occurred.';
      if (err instanceof Error) {
          if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
              message = 'Microphone permission denied. Please allow microphone access in your browser settings.';
          } else {
              message = err.message;
          }
      }
      setError(message);
      setIsConnecting(false);
    }
  }, [isListening, isConnecting, stopListening, currentTranscript]);

  const handleTranscribeFile = useCallback(async () => {
    if (!selectedFile) return;

    setFileProcessingState('processing');
    setFileTranscript('');
    setFileError(null);

    try {
        const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
        const arrayBuffer = await selectedFile.arrayBuffer();
        const decodedBuffer = await audioContext.decodeAudioData(arrayBuffer);

        const targetSampleRate = 16000;
        const offlineContext = new OfflineAudioContext(1, decodedBuffer.duration * targetSampleRate, targetSampleRate);
        const source = offlineContext.createBufferSource();
        source.buffer = decodedBuffer;
        source.connect(offlineContext.destination);
        source.start();
        const resampledBuffer = await offlineContext.startRendering();
        const pcmData = resampledBuffer.getChannelData(0);

        const ai = new GoogleGenAI({ apiKey: process.env.API_KEY as string });
        let accumulatedTranscript = '';
        
        const sessionPromise = ai.live.connect({
            model: 'gemini-2.5-flash-native-audio-preview-09-2025',
            callbacks: {
                onopen: async () => {
                    const chunkSize = 4096;
                    for (let i = 0; i < pcmData.length; i += chunkSize) {
                        const chunk = pcmData.slice(i, i + chunkSize);
                        if (chunk.length === 0) continue;

                        const pcmBlob = createBlob(chunk);
                        try {
                          const session = await sessionPromise;
                          session.sendRealtimeInput({ media: pcmBlob });
                        } catch(e) {
                          console.error("Failed to send chunk", e);
                          // Stop sending more chunks on error
                          break;
                        }
                        await new Promise(resolve => setTimeout(resolve, 100));
                    }
                    const session = await sessionPromise;
                    session.close();
                },
                onmessage: (message: LiveServerMessage) => {
                    if (message.serverContent?.inputTranscription) {
                        const text = message.serverContent.inputTranscription.text;
                        accumulatedTranscript += text;
                        setFileTranscript(prev => prev + text);
                    }
                },
                onerror: (e: ErrorEvent) => {
                    console.error("Session error:", e);
                    setFileError(`Transcription failed: ${e.message}.`);
                    setFileProcessingState('error');
                },
                onclose: () => {
                    setFileTranscript(accumulatedTranscript.trim());
                    setFileProcessingState('success');
                },
            },
            config: { 
                inputAudioTranscription: {},
                responseModalities: [Modality.AUDIO],
            },
        });
        
    } catch (err) {
        console.error("Failed to transcribe file:", err);
        let message = 'File processing failed. It may not be a valid audio file.';
        if (err instanceof Error) message = err.message;
        setFileError(message);
        setFileProcessingState('error');
    }
  }, [selectedFile]);

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      if (file.type === "audio/mpeg") {
        setSelectedFile(file);
        setFileTranscript('');
        setFileProcessingState('idle');
        setFileError(null);
      } else {
        setFileError("Invalid file type. Please select an MP3 file.");
        setSelectedFile(null);
      }
    }
  };

  const handleCopyToClipboard = () => {
    if(fileTranscript) {
      navigator.clipboard.writeText(fileTranscript);
      setIsCopied(true);
      setTimeout(() => setIsCopied(false), 2000);
    }
  };

  const ModeButton = ({ active, onClick, children }: { active: boolean, onClick: () => void, children: React.ReactNode }) => (
    <button onClick={onClick} className={`w-1/2 py-2 px-4 text-sm font-bold rounded-md transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-cyan-400 focus:ring-opacity-50 ${active ? 'bg-cyan-500 text-white' : 'text-slate-300 hover:bg-slate-600/50'}`}>
        {children}
    </button>
  );

  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-4 bg-slate-900 font-sans">
      <div className="w-full max-w-3xl bg-slate-800 rounded-2xl shadow-2xl flex flex-col overflow-hidden ring-1 ring-slate-700">
        <header className="p-6 border-b border-slate-700">
          <h1 className="text-3xl font-bold text-center text-cyan-400">
            Audio Transcriber
          </h1>
          <p className="text-center text-slate-400 mt-2">
            Transcribe spoken words from your microphone or an audio file into English text.
          </p>
        </header>

        <div className="flex p-1 bg-slate-700/50 rounded-lg mx-6 mt-6">
            <ModeButton active={mode === 'live'} onClick={() => setMode('live')}>Live Transcription</ModeButton>
            <ModeButton active={mode === 'file'} onClick={() => setMode('file')}>File Transcription</ModeButton>
        </div>

        <main className="flex-grow p-6 flex flex-col gap-6">
            <div className="relative w-full h-64 bg-slate-900 rounded-lg p-4 overflow-y-auto ring-1 ring-slate-700 focus-within:ring-2 focus-within:ring-cyan-500 transition-shadow">
                {mode === 'live' ? (
                  <>
                    {transcriptHistory.map((text, index) => (
                        <p key={index} className="text-slate-300 mb-2">{text}</p>
                    ))}
                    <p className="text-white font-medium">{currentTranscript}<span className="animate-pulse">_</span></p>
                    {transcriptHistory.length === 0 && currentTranscript === '' && !isListening && (
                        <div className="absolute inset-0 flex items-center justify-center">
                            <p className="text-slate-500">Live transcript will appear here...</p>
                        </div>
                    )}
                  </>
                ) : (
                  <>
                    {fileProcessingState === 'success' && fileTranscript && (
                      <button onClick={handleCopyToClipboard} className="absolute top-2 right-2 p-2 rounded-md bg-slate-700 hover:bg-slate-600 transition-colors text-slate-300 hover:text-white">
                        {isCopied ? <span className="text-xs">Copied!</span> : <ClipboardIcon className="w-5 h-5" />}
                      </button>
                    )}
                    <p className="text-white font-medium whitespace-pre-wrap">{fileTranscript}</p>
                    {fileProcessingState === 'idle' && !fileTranscript && (
                      <div className="absolute inset-0 flex items-center justify-center">
                          <p className="text-slate-500">File transcript will appear here...</p>
                      </div>
                    )}
                     {fileProcessingState === 'processing' && !fileTranscript && (
                      <div className="absolute inset-0 flex items-center justify-center flex-col gap-2">
                          <LoadingSpinnerIcon className="w-8 h-8 animate-spin text-cyan-400"/>
                          <p className="text-slate-400">Transcribing, please wait...</p>
                      </div>
                    )}
                  </>
                )}
            </div>

            {(error && mode === 'live') && (
                <div className="bg-red-900/50 text-red-300 p-4 rounded-lg flex items-center gap-3 ring-1 ring-red-800">
                    <WarningIcon className="w-6 h-6 flex-shrink-0"/>
                    <div><h3 className="font-bold">An Error Occurred</h3><p className="text-sm">{error}</p></div>
                </div>
            )}
             {(fileError && mode === 'file') && (
                <div className="bg-red-900/50 text-red-300 p-4 rounded-lg flex items-center gap-3 ring-1 ring-red-800">
                    <WarningIcon className="w-6 h-6 flex-shrink-0"/>
                    <div><h3 className="font-bold">An Error Occurred</h3><p className="text-sm">{fileError}</p></div>
                </div>
            )}
        </main>
        
        <footer className="p-6 border-t border-slate-700 flex flex-col items-center justify-center gap-4">
          {mode === 'live' ? (
            <>
              <button onClick={isListening ? stopListening : startListening} disabled={isConnecting}
                className={`flex items-center justify-center gap-3 px-8 py-4 w-56 rounded-full text-lg font-semibold transition-all duration-300 ease-in-out focus:outline-none focus:ring-4 focus:ring-opacity-50 ${isListening ? 'bg-red-600 hover:bg-red-700 text-white focus:ring-red-500' : 'bg-cyan-500 hover:bg-cyan-600 text-white focus:ring-cyan-400'} ${isConnecting ? 'bg-slate-600 cursor-not-allowed' : ''} shadow-lg hover:shadow-xl transform hover:scale-105`}>
                {isConnecting ? (<><LoadingSpinnerIcon className="w-6 h-6 animate-spin" />Connecting...</>) 
                : isListening ? (<><StopIcon className="w-6 h-6" />Stop Listening</>) 
                : (<><MicrophoneIcon className="w-6 h-6" />Start Listening</>)}
              </button>
              <p className="text-slate-500 text-sm h-5">{isConnecting ? 'Initializing session...' : isListening ? 'Actively listening to your microphone...' : 'Ready to transcribe.'}</p>
            </>
          ) : (
             <>
                <div className="w-full max-w-sm flex flex-col items-center gap-4">
                  <label htmlFor="file-upload" className="w-full relative flex flex-col items-center justify-center p-6 border-2 border-dashed border-slate-600 rounded-lg cursor-pointer hover:bg-slate-700/50 transition-colors">
                    <UploadCloudIcon className="w-10 h-10 text-slate-500 mb-2"/>
                    <p className="text-slate-400 text-sm"><span className="font-semibold text-cyan-400">Click to upload</span> or drag and drop</p>
                    <p className="text-slate-500 text-xs">MP3 files only</p>
                    <input id="file-upload" type="file" className="hidden" accept="audio/mpeg" onChange={handleFileChange} disabled={fileProcessingState === 'processing'}/>
                  </label>
                  {selectedFile && <p className="text-sm text-slate-300 flex items-center gap-2"><FileMusicIcon className="w-4 h-4"/>{selectedFile.name}</p>}
                </div>
                <button onClick={handleTranscribeFile} disabled={!selectedFile || fileProcessingState === 'processing'}
                    className="flex items-center justify-center gap-3 px-8 py-4 w-56 rounded-full text-lg font-semibold transition-all duration-300 ease-in-out focus:outline-none focus:ring-4 focus:ring-opacity-50 bg-cyan-500 hover:bg-cyan-600 text-white focus:ring-cyan-400 disabled:bg-slate-600 disabled:cursor-not-allowed shadow-lg hover:shadow-xl transform hover:scale-105 disabled:transform-none disabled:shadow-none">
                    {fileProcessingState === 'processing' ? (<><LoadingSpinnerIcon className="w-6 h-6 animate-spin" />Transcribing...</>) : 'Transcribe File'}
                </button>
             </>
          )}
        </footer>
      </div>
      <p className="text-slate-600 text-sm mt-6">Powered by Gemini API</p>
    </div>
  );
}