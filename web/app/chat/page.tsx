'use client'

import { useState, useEffect } from 'react'
import { useSearchParams } from 'next/navigation'

export default function ChatPage() {
  const [messages, setMessages] = useState<Array<{role: 'user' | 'assistant', content: string}>>([])
  const [input, setInput] = useState('')
  const [url, setUrl] = useState('')
  const [loading, setLoading] = useState(false)
  const searchParams = useSearchParams()

  useEffect(() => {
    const urlParam = searchParams.get('url')
    if (urlParam) {
      setUrl(urlParam)
      setMessages([{role: 'assistant', content: `I'm ready to chat about: ${urlParam}`}])
    }
  }, [searchParams])

  async function sendMessage(e: React.FormEvent) {
    e.preventDefault()
    if (!input.trim()) return

    const userMessage = input.trim()
    setInput('')
    setMessages(prev => [...prev, {role: 'user', content: userMessage}])
    setLoading(true)

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({message: userMessage, url})
      })
      
      const data = await response.json()
      setMessages(prev => [...prev, {role: 'assistant', content: data.response || 'Sorry, something went wrong'}])
    } catch (error) {
      setMessages(prev => [...prev, {role: 'assistant', content: 'Error: Could not get response'}])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 p-4">
      <div className="max-w-2xl mx-auto">
        <div className="bg-white rounded-lg shadow p-6 mb-4">
          <h1 className="text-2xl font-bold mb-2">Chat</h1>
          <p className="text-gray-600 text-sm">URL: {url}</p>
        </div>

        <div className="bg-white rounded-lg shadow p-6 h-96 overflow-y-auto mb-4">
          {messages.map((msg, i) => (
            <div key={i} className={`mb-4 p-3 rounded ${
              msg.role === 'user' ? 'bg-blue-100 ml-8' : 'bg-gray-100 mr-8'
            }`}>
              <div className="font-semibold text-sm mb-1">
                {msg.role === 'user' ? 'You' : 'Assistant'}
              </div>
              <div>{msg.content}</div>
            </div>
          ))}
          {loading && (
            <div className="text-gray-500 text-center">Thinking...</div>
          )}
        </div>

        <form onSubmit={sendMessage} className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask something about this URL..."
            className="flex-1 px-3 py-2 border border-gray-300 rounded-md"
            disabled={loading}
          />
          <button
            type="submit"
            disabled={loading || !input.trim()}
            className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 disabled:opacity-50"
          >
            Send
          </button>
        </form>
      </div>
    </div>
  )
} 