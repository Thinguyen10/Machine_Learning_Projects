// Component to display sentiment analysis results
export default function SentimentCard({ sentiment }) {
  return (
    <div className="sentiment-card">
      <h2>Analysis Result</h2>
      <p>Sentiment: {sentiment.label}</p>
      <p>Confidence: {(sentiment.score * 100).toFixed(2)}%</p>
    </div>
  );
}
