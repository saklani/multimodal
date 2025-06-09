'use client';

import React, { useState, useRef, useEffect } from 'react';
import { Send, Youtube, Play } from 'lucide-react';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  citations?: Citation[];
}

interface Citation {
  text: string;
  timestamp_seconds: number;
  timestamp_formatted: string;
}

interface Section {
  title: string;
  timestamp_seconds: number;
  timestamp_formatted: string;
  text_segment: string;
}

interface VisualFrame {
  timestamp_seconds: number;
  timestamp_formatted: string;
  description: string;
}

interface VisualSearchResult {
  timestamp_seconds: number;
  timestamp_formatted: string;
  description: string;
  youtube_url: string;
}

interface VideoData {
  video_id: string;
  title: string;
  sections: Section[];
  full_transcript: string;
  visual_frames: VisualFrame[];
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [youtubeUrl, setYoutubeUrl] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [videoData, setVideoData] = useState<VideoData | null>(null);
  const [visualSearchQuery, setVisualSearchQuery] = useState('');
  const [visualSearchResults, setVisualSearchResults] = useState<VisualSearchResult[]>([]);
  const [isVisualSearchLoading, setIsVisualSearchLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleVideoProcess = async () => {
    if (!youtubeUrl.trim()) return;

    setIsLoading(true);
    try {
      const response = await fetch('http://localhost:8000/process_video', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ youtube_url: youtubeUrl }),
      });

      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.detail || 'Failed to process video');
      }

      setVideoData(data);
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: `Successfully processed video "${data.title}". Found ${data.sections.length} segments. You can now ask questions about the video content!` 
      }]);
    } catch (error) {
      console.error('Error:', error);
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: `Sorry, there was an error processing the video: ${error instanceof Error ? error.message : 'Unknown error'}` 
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || !videoData) return;

    const userMessage = input.trim();
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
    setIsLoading(true);

    try {
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video_id: videoData.video_id, query: userMessage }),
      });

      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.detail || 'Failed to get response');
      }

      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: data.answer,
        citations: data.citations
      }]);
    } catch (error) {
      console.error('Error:', error);
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: `Sorry, there was an error processing your question: ${error instanceof Error ? error.message : 'Unknown error'}` 
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const formatYouTubeUrl = (videoId: string, timestamp?: number) => {
    const baseUrl = `https://www.youtube.com/watch?v=${videoId}`;
    return timestamp ? `${baseUrl}&t=${Math.floor(timestamp)}s` : baseUrl;
  };

  const handleVisualSearch = async () => {
    if (!visualSearchQuery.trim() || !videoData) return;

    setIsVisualSearchLoading(true);
    try {
      const response = await fetch('http://localhost:8000/visual_search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          video_id: videoData.video_id, 
          query: visualSearchQuery 
        }),
      });

      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.detail || 'Failed to perform visual search');
      }

      setVisualSearchResults(data.results || []);
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: data.message || `Found ${data.results?.length || 0} matching visual segments.` 
      }]);
    } catch (error) {
      console.error('Error:', error);
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: `Sorry, there was an error with the visual search: ${error instanceof Error ? error.message : 'Unknown error'}` 
      }]);
      setVisualSearchResults([]);
    } finally {
      setIsVisualSearchLoading(false);
    }
  };

  return (
    <main className="flex min-h-screen flex-col items-center p-2 bg-gray-100">
      <div className="w-full max-w-6xl flex flex-col h-screen">
        <h1 className="text-lg font-bold text-center mb-4 text-gray-900">
          YouTube Video Q&A
        </h1>
        
        {/* Video URL Input */}
        <div className="mb-3 bg-white rounded-xs border border-gray-300 p-2">
          <div className="flex gap-2">
            <input
              type="url"
              value={youtubeUrl}
              onChange={(e) => setYoutubeUrl(e.target.value)}
              placeholder="Enter YouTube URL (e.g., https://www.youtube.com/watch?v=...)"
              className="flex-1 p-2 border border-gray-400 rounded-xs bg-white text-gray-900 text-sm"
              disabled={isLoading}
            />
            <button
              onClick={handleVideoProcess}
              disabled={isLoading || !youtubeUrl.trim()}
              className="px-3 py-2 bg-gray-900 text-white rounded-xs hover:bg-black disabled:opacity-50 flex items-center gap-1 text-sm"
            >
              <Youtube className="w-4 h-4" />
              Process Video
            </button>
          </div>
        </div>

        <div className="flex-1 flex gap-3">
          {/* Left Sidebar */}
          {videoData && (
            <div className="w-1/3 flex flex-col gap-3">
              {/* Video Sections Panel */}
              <div className="bg-white rounded-xs border border-gray-300 p-2 overflow-y-auto flex-1">
                <h2 className="text-sm font-semibold mb-2 text-gray-900">Video Segments ({videoData.sections.length})</h2>
                <div className="space-y-1 max-h-80 overflow-y-auto">
                  {videoData.sections.map((section, index) => (
                    <div
                      key={index}
                      className="p-2 border border-gray-200 rounded-xs hover:bg-gray-50 cursor-pointer"
                      onClick={() => window.open(formatYouTubeUrl(videoData.video_id, section.timestamp_seconds), '_blank')}
                    >
                      <div className="flex items-center gap-2 text-xs">
                        <Play className="w-3 h-3 text-gray-800" />
                        <span className="font-mono text-gray-800">{section.timestamp_formatted}</span>
                      </div>
                      <p className="text-xs text-gray-700 mt-1 line-clamp-2">{section.text_segment}</p>
                    </div>
                  ))}
                </div>
              </div>

              {/* Visual Search Panel */}
              <div className="bg-white rounded-xs border border-gray-300 p-2">
                <h2 className="text-sm font-semibold mb-2 text-gray-900">Visual Search</h2>
                <div className="space-y-2">
                  <div className="flex gap-2">
                    <input
                      type="text"
                      value={visualSearchQuery}
                      onChange={(e) => setVisualSearchQuery(e.target.value)}
                      placeholder="Find visual content (e.g., 'red car', 'person speaking')"
                      className="flex-1 p-2 border border-gray-400 rounded-xs bg-white text-gray-900 text-xs"
                      disabled={isVisualSearchLoading}
                    />
                    <button
                      onClick={handleVisualSearch}
                      disabled={isVisualSearchLoading || !visualSearchQuery.trim()}
                      className="px-3 py-2 bg-gray-900 text-white rounded-xs hover:bg-black disabled:opacity-50 text-xs"
                    >
                      Search
                    </button>
                  </div>
                  
                  {visualSearchResults.length > 0 && (
                    <div className="space-y-1 max-h-32 overflow-y-auto">
                      <p className="text-xs text-gray-800 font-medium">Visual Results:</p>
                      {visualSearchResults.map((result, index) => (
                        <div
                          key={index}
                          className="p-2 border border-gray-200 rounded-xs hover:bg-gray-50 cursor-pointer"
                          onClick={() => window.open(result.youtube_url, '_blank')}
                        >
                          <div className="flex items-center gap-2 text-xs">
                            <Play className="w-3 h-3 text-gray-800" />
                            <span className="font-mono text-gray-800">{result.timestamp_formatted}</span>
                          </div>
                          <p className="text-xs text-gray-700 mt-1 line-clamp-2">{result.description}</p>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Chat Panel */}
          <div className="flex-1 flex flex-col">
            <div className="flex-1 overflow-y-auto mb-2 bg-white rounded-xs border border-gray-300 p-2">
              {messages.map((message, index) => (
                <div
                  key={index}
                  className={`mb-2 p-2 rounded-xs text-sm ${
                    message.role === 'user'
                      ? 'bg-gray-200 ml-auto max-w-[80%]'
                    : 'bg-gray-100 mr-auto max-w-[80%]'
                  }`}
                >
                  <p className="text-gray-900 whitespace-pre-wrap">{message.content}</p>
                  {message.citations && message.citations.length > 0 && (
                    <div className="mt-2 pt-2 border-t border-gray-300">
                      <p className="text-xs text-gray-800 mb-1">Citations:</p>
                      <div className="space-y-1">
                        {message.citations.map((citation, citIndex) => (
                          <div key={citIndex} className="text-xs">
                            <button
                              onClick={() => window.open(formatYouTubeUrl(videoData?.video_id || '', citation.timestamp_seconds), '_blank')}
                              className="text-gray-800 hover:underline font-mono"
                            >
                              [{citation.timestamp_formatted}]
                            </button>
                            <span className="text-gray-700 ml-2">{citation.text}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ))}
              <div ref={messagesEndRef} />
            </div>

            <form onSubmit={handleSubmit} className="flex gap-2">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder={videoData ? "Ask a question about the video..." : "Process a video first to start asking questions"}
                className="flex-1 p-2 border border-gray-400 rounded-xs bg-white text-gray-900 text-sm"
                disabled={isLoading || !videoData}
              />
              <button
                type="submit"
                disabled={isLoading || !input.trim() || !videoData}
                className="px-3 py-2 bg-gray-900 text-white rounded-xs hover:bg-black disabled:opacity-50"
              >
                <Send className="w-4 h-4" />
              </button>
            </form>
          </div>
        </div>
      </div>
    </main>
  );
}
