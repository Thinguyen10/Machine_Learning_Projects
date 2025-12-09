// Dashboard data API for Vercel deployment
export default async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  if (req.method !== 'POST' && req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  // Check if we have real batch results
  const batchResults = global.lastBatchResults || [];
  
  if (batchResults.length > 0) {
    // Calculate real statistics from uploaded data
    const positiveCount = batchResults.filter(r => r.sentiment === 'positive').length;
    const negativeCount = batchResults.filter(r => r.sentiment === 'negative').length;
    const neutralCount = batchResults.length - positiveCount - negativeCount;
    const total = batchResults.length;

    return res.status(200).json({
      statistics: {
        total_reviews: total,
        positive_count: positiveCount,
        negative_count: negativeCount,
        neutral_count: neutralCount,
        positive_percentage: ((positiveCount / total) * 100).toFixed(2),
        negative_percentage: ((negativeCount / total) * 100).toFixed(2),
        neutral_percentage: ((neutralCount / total) * 100).toFixed(2)
      },
      trends: [],
      top_aspects: [],
      reviews: batchResults.slice(0, 10),
      data_source: 'uploaded_batch',
      timestamp: new Date().toISOString()
    });
  }

  // Return demo data if no uploads yet
  const demoData = {
    statistics: {
      total_reviews: 245,
      positive_count: 142,
      negative_count: 58,
      neutral_count: 45,
      positive_percentage: 57.96,
      negative_percentage: 23.67,
      neutral_percentage: 18.37
    },
    trends: [
      {
        date: '2024-12-01',
        positive_count: 18,
        negative_count: 5,
        neutral_count: 7,
        total_count: 30,
        positive_pct: 60.0,
        negative_pct: 16.7,
        neutral_pct: 23.3
      },
      {
        date: '2024-12-02',
        positive_count: 22,
        negative_count: 8,
        neutral_count: 5,
        total_count: 35,
        positive_pct: 62.9,
        negative_pct: 22.9,
        neutral_pct: 14.3
      },
      {
        date: '2024-12-03',
        positive_count: 25,
        negative_count: 10,
        neutral_count: 8,
        total_count: 43,
        positive_pct: 58.1,
        negative_pct: 23.3,
        neutral_pct: 18.6
      },
      {
        date: '2024-12-04',
        positive_count: 20,
        negative_count: 12,
        neutral_count: 6,
        total_count: 38,
        positive_pct: 52.6,
        negative_pct: 31.6,
        neutral_pct: 15.8
      },
      {
        date: '2024-12-05',
        positive_count: 28,
        negative_count: 9,
        neutral_count: 8,
        total_count: 45,
        positive_pct: 62.2,
        negative_pct: 20.0,
        neutral_pct: 17.8
      },
      {
        date: '2024-12-06',
        positive_count: 29,
        negative_count: 14,
        neutral_count: 11,
        total_count: 54,
        positive_pct: 53.7,
        negative_pct: 25.9,
        neutral_pct: 20.4
      }
    ],
    aspects: [
      {
        aspect_name: 'quality',
        dominant_sentiment: 'positive',
        positive_mentions: 45,
        negative_mentions: 12,
        neutral_mentions: 8,
        total_mentions: 65
      },
      {
        aspect_name: 'price',
        dominant_sentiment: 'negative',
        positive_mentions: 18,
        negative_mentions: 32,
        neutral_mentions: 5,
        total_mentions: 55
      },
      {
        aspect_name: 'service',
        dominant_sentiment: 'positive',
        positive_mentions: 38,
        negative_mentions: 15,
        neutral_mentions: 7,
        total_mentions: 60
      },
      {
        aspect_name: 'shipping',
        dominant_sentiment: 'neutral',
        positive_mentions: 20,
        negative_mentions: 18,
        neutral_mentions: 12,
        total_mentions: 50
      },
      {
        aspect_name: 'packaging',
        dominant_sentiment: 'positive',
        positive_mentions: 25,
        negative_mentions: 8,
        neutral_mentions: 5,
        total_mentions: 38
      }
    ],
    recent_batches: [
      {
        id: 1,
        filename: 'customer_reviews.csv',
        upload_timestamp: '2024-12-08T10:30:00',
        total_rows: 245,
        processed_rows: 245,
        status: 'completed'
      }
    ],
    note: 'Demo data - upload CSV files to analyze your own data'
  };

  return res.status(200).json(demoData);
}
