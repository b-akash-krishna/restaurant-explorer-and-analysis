import { useState, useEffect } from 'react';
import { Utensils, Loader2, RefreshCw, TrendingUp, AlertCircle } from 'lucide-react';

interface ClassificationResult {
  predicted_cuisine: string;
  confidence: number;
  top_predictions: Array<{
    cuisine: string;
    confidence: number;
  }>;
}

interface ClassificationOptions {
  cities: string[];
  price_ranges: Record<string, string>;
  random_sample: {
    city: string;
    has_table_booking: number;
    has_online_delivery: number;
    price_range: number;
    votes: number;
  };
  votes_range: {
    min: number;
    max: number;
    avg: number;
  };
}

interface ModelPerformance {
  test_metrics: {
    accuracy: number;
    precision_macro: number;
    recall_macro: number;
    f1_macro: number;
  };
  per_cuisine_metrics: Record<string, {
    precision: number;
    recall: number;
    'f1-score': number;
    support: number;
  }>;
  total_cuisines: number;
  cuisines: string[];
}

function CuisineClassification() {
  const [loading, setLoading] = useState(false);
  const [optionsLoading, setOptionsLoading] = useState(true);
  const [result, setResult] = useState<ClassificationResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [options, setOptions] = useState<ClassificationOptions | null>(null);
  const [performance, setPerformance] = useState<ModelPerformance | null>(null);
  const [showPerformance, setShowPerformance] = useState(false);

  const [formData, setFormData] = useState({
    city: '',
    has_table_booking: 1,
    has_online_delivery: 1,
    price_range: 2,
    votes: 150,
  });

  useEffect(() => {
    fetchOptions();
    fetchPerformance();
  }, []);

  const fetchOptions = async () => {
    setOptionsLoading(true);
    try {
      const response = await fetch('http://localhost:8000/api/classify-cuisine/options');
      if (response.ok) {
        const data = await response.json();
        setOptions(data);
        setFormData(data.random_sample);
      }
    } catch (err: any) {
      setError('Failed to load options');
    } finally {
      setOptionsLoading(false);
    }
  };

  const fetchPerformance = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/classify-cuisine/performance');
      if (response.ok) {
        const data = await response.json();
        setPerformance(data);
      }
    } catch (err) {
      console.error('Failed to load performance metrics');
    }
  };

  const loadRandomSample = () => {
    fetchOptions();
    setResult(null);
    setError(null);
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: ['has_table_booking', 'has_online_delivery', 'price_range', 'votes'].includes(name)
        ? Number(value)
        : value,
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch('http://localhost:8000/api/classify-cuisine', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData),
      });

      if (!response.ok) throw new Error('Classification failed');

      const data = await response.json();
      setResult(data);
    } catch (err: any) {
      setError(err.message || 'Failed to classify cuisine');
    } finally {
      setLoading(false);
    }
  };

  if (optionsLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-orange-50 via-amber-50 to-yellow-50 flex items-center justify-center p-4">
        <div className="bg-white/95 backdrop-blur-md shadow-2xl rounded-2xl p-8 w-full max-w-4xl border border-orange-200">
          <div className="flex items-center justify-center mb-6">
            <Loader2 className="w-12 h-12 text-orange-500 animate-spin" />
          </div>
          <p className="text-center text-gray-600">Loading classification options...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-orange-50 via-amber-50 to-yellow-50">
      {/* Hero Header */}
      <div 
        className="relative bg-cover bg-center py-16 mb-8"
        style={{
          backgroundImage: "url('https://images.unsplash.com/photo-1504674900247-0877df9cc836?w=1600&h=400&fit=crop')",
        }}
      >
        <div className="absolute inset-0 bg-gradient-to-r from-orange-900/90 via-amber-800/85 to-yellow-900/90"></div>
        <div className="relative max-w-7xl mx-auto px-4">
          <div className="flex items-center justify-center mb-4">
            <div className="bg-white/20 backdrop-blur-md p-4 rounded-2xl">
              <Utensils className="w-12 h-12 text-white" />
            </div>
          </div>
          <h1 className="text-4xl md:text-5xl font-bold text-center text-white mb-3 drop-shadow-lg">
            Cuisine Classification
          </h1>
          <p className="text-center text-orange-100 text-lg max-w-2xl mx-auto drop-shadow-md">
            Multi-class classification using Random Forest to predict restaurant cuisine types
          </p>
        </div>
      </div>

      <div className="max-w-6xl mx-auto px-4 pb-8">
        {/* Control Panel */}
        <div className="bg-white/95 backdrop-blur-sm shadow-xl rounded-2xl p-6 mb-6 border border-orange-100">
          <div className="flex justify-center gap-4">
            <button
              onClick={loadRandomSample}
              className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-gray-100 to-gray-200 hover:from-gray-200 hover:to-gray-300 text-gray-700 rounded-xl transition-all duration-300 font-medium shadow-md hover:shadow-lg transform hover:-translate-y-0.5"
              disabled={loading}
            >
              <RefreshCw className="w-4 h-4" />
              Load Random Sample
            </button>
            <button
              onClick={() => setShowPerformance(!showPerformance)}
              className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-orange-500 to-amber-500 hover:from-orange-600 hover:to-amber-600 text-white rounded-xl transition-all duration-300 font-medium shadow-md hover:shadow-lg transform hover:-translate-y-0.5"
            >
              <TrendingUp className="w-4 h-4" />
              {showPerformance ? 'Hide' : 'Show'} Model Performance
            </button>
          </div>
        </div>

        {/* Model Performance Section */}
        {showPerformance && performance && (
          <div className="mb-6 p-8 bg-white/95 backdrop-blur-sm border-2 border-orange-200 rounded-2xl shadow-xl">
            <h3 className="text-2xl font-bold text-gray-900 mb-6 flex items-center gap-2">
              <TrendingUp className="w-6 h-6 text-orange-600" />
              Model Performance Metrics
            </h3>
            
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6 mb-8">
              <div className="bg-gradient-to-br from-orange-50 to-orange-100 p-6 rounded-xl shadow-inner">
                <div className="text-3xl font-bold text-orange-600">
                  {(performance.test_metrics.accuracy * 100).toFixed(1)}%
                </div>
                <div className="text-sm text-gray-600 mt-1 font-medium">Accuracy</div>
              </div>
              <div className="bg-gradient-to-br from-blue-50 to-blue-100 p-6 rounded-xl shadow-inner">
                <div className="text-3xl font-bold text-blue-600">
                  {(performance.test_metrics.precision_macro * 100).toFixed(1)}%
                </div>
                <div className="text-sm text-gray-600 mt-1 font-medium">Precision</div>
              </div>
              <div className="bg-gradient-to-br from-green-50 to-green-100 p-6 rounded-xl shadow-inner">
                <div className="text-3xl font-bold text-green-600">
                  {(performance.test_metrics.recall_macro * 100).toFixed(1)}%
                </div>
                <div className="text-sm text-gray-600 mt-1 font-medium">Recall</div>
              </div>
              <div className="bg-gradient-to-br from-purple-50 to-purple-100 p-6 rounded-xl shadow-inner">
                <div className="text-3xl font-bold text-purple-600">
                  {(performance.test_metrics.f1_macro * 100).toFixed(1)}%
                </div>
                <div className="text-sm text-gray-600 mt-1 font-medium">F1-Score</div>
              </div>
            </div>

            <div className="bg-white/80 backdrop-blur-sm p-6 rounded-xl border border-orange-100">
              <h4 className="font-bold text-gray-900 mb-4 text-lg">Per-Cuisine Performance</h4>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b-2 border-orange-200">
                      <th className="text-left py-3 px-3 font-bold">Cuisine</th>
                      <th className="text-right py-3 px-3 font-bold">Precision</th>
                      <th className="text-right py-3 px-3 font-bold">Recall</th>
                      <th className="text-right py-3 px-3 font-bold">F1-Score</th>
                      <th className="text-right py-3 px-3 font-bold">Support</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(performance.per_cuisine_metrics).slice(0, 10).map(([cuisine, metrics]) => (
                      <tr key={cuisine} className="border-b border-gray-200 hover:bg-orange-50 transition-colors">
                        <td className="py-3 px-3 font-semibold">{cuisine}</td>
                        <td className="text-right py-3 px-3">{(metrics.precision * 100).toFixed(1)}%</td>
                        <td className="text-right py-3 px-3">{(metrics.recall * 100).toFixed(1)}%</td>
                        <td className="text-right py-3 px-3">
                          <span className={metrics['f1-score'] < 0.7 ? 'text-red-600 font-bold' : 'font-medium'}>
                            {(metrics['f1-score'] * 100).toFixed(1)}%
                          </span>
                        </td>
                        <td className="text-right py-3 px-3 font-medium">{metrics.support}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="text-xs text-gray-500 mt-4 flex items-center">
                <AlertCircle className="w-4 h-4 inline mr-2" />
                Red F1-scores indicate cuisines with classification challenges (potential biases or low support)
              </p>
            </div>
          </div>
        )}

        {/* Classification Form */}
        <div className="bg-white/95 backdrop-blur-sm shadow-xl rounded-2xl p-8 border border-orange-100">
          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">
                  City
                </label>
                <select
                  name="city"
                  value={formData.city}
                  onChange={handleChange}
                  className="w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:ring-4 focus:ring-orange-100 focus:border-orange-500 transition-all duration-200"
                  required
                >
                  {options?.cities.map((city, index) => (
                    <option key={index} value={city}>{city}</option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">
                  Number of Votes
                </label>
                <input
                  type="number"
                  name="votes"
                  value={formData.votes}
                  onChange={handleChange}
                  min={options?.votes_range.min || 0}
                  max={options?.votes_range.max || 10000}
                  className="w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:ring-4 focus:ring-orange-100 focus:border-orange-500 transition-all duration-200"
                  required
                />
                <p className="text-xs text-gray-500 mt-2">
                  Range: {options?.votes_range.min} - {options?.votes_range.max}
                </p>
              </div>

              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">
                  Price Range
                </label>
                <select
                  name="price_range"
                  value={formData.price_range}
                  onChange={handleChange}
                  className="w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:ring-4 focus:ring-orange-100 focus:border-orange-500 transition-all duration-200"
                  required
                >
                  {options?.price_ranges && Object.entries(options.price_ranges).map(([value, label]) => (
                    <option key={value} value={value}>{label}</option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">
                  Table Booking
                </label>
                <select
                  name="has_table_booking"
                  value={formData.has_table_booking}
                  onChange={handleChange}
                  className="w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:ring-4 focus:ring-orange-100 focus:border-orange-500 transition-all duration-200"
                >
                  <option value="1">Yes</option>
                  <option value="0">No</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">
                  Online Delivery
                </label>
                <select
                  name="has_online_delivery"
                  value={formData.has_online_delivery}
                  onChange={handleChange}
                  className="w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:ring-4 focus:ring-orange-100 focus:border-orange-500 transition-all duration-200"
                >
                  <option value="1">Yes</option>
                  <option value="0">No</option>
                </select>
              </div>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full bg-gradient-to-r from-orange-600 to-amber-600 text-white py-4 rounded-xl font-semibold hover:from-orange-700 hover:to-amber-700 transition-all duration-300 disabled:from-orange-300 disabled:to-amber-300 disabled:cursor-not-allowed flex items-center justify-center shadow-lg hover:shadow-xl transform hover:-translate-y-0.5"
            >
              {loading ? (
                <>
                  <Loader2 className="animate-spin h-5 w-5 mr-2" />
                  Classifying...
                </>
              ) : (
                'Classify Cuisine'
              )}
            </button>
          </form>

          {error && (
            <div className="mt-6 p-4 bg-red-50 border-2 border-red-200 rounded-xl">
              <p className="text-red-600 text-sm font-medium">{error}</p>
            </div>
          )}

          {result && (
            <div className="mt-6 p-8 bg-gradient-to-br from-orange-50 to-yellow-50 border-2 border-orange-200 rounded-2xl shadow-lg">
              <h3 className="text-xl font-bold text-gray-900 mb-6 flex items-center gap-2">
                <Utensils className="w-6 h-6 text-orange-600" />
                Classification Result
              </h3>
              
              <div className="text-center mb-8">
                <div className="bg-white/80 backdrop-blur-sm p-8 rounded-2xl shadow-md inline-block">
                  <div className="text-5xl font-bold bg-gradient-to-r from-orange-600 to-amber-600 bg-clip-text text-transparent mb-2">
                    {result.predicted_cuisine}
                  </div>
                  <div className="text-sm text-gray-600 font-medium">Predicted Cuisine Type</div>
                </div>
              </div>

              <div className="mb-8 bg-white/60 backdrop-blur-sm p-4 rounded-xl">
                <div className="flex items-center justify-between mb-3">
                  <span className="text-sm font-bold text-gray-700">Confidence Level</span>
                  <span className="text-base font-bold text-orange-600">{result.confidence.toFixed(1)}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-4">
                  <div
                    className="bg-gradient-to-r from-orange-500 to-yellow-500 h-4 rounded-full transition-all duration-500 shadow-md"
                    style={{ width: `${result.confidence}%` }}
                  ></div>
                </div>
              </div>

              {result.top_predictions && result.top_predictions.length > 1 && (
                <div className="mt-8 pt-6 border-t-2 border-orange-200">
                  <p className="text-sm font-bold text-gray-700 mb-4">Top 3 Predictions:</p>
                  <div className="space-y-3">
                    {result.top_predictions.map((pred, index) => (
                      <div key={index} className="flex items-center justify-between bg-white/80 backdrop-blur-sm p-4 rounded-xl shadow-sm hover:shadow-md transition-all">
                        <div className="flex items-center gap-3">
                          <div className={`w-10 h-10 rounded-full flex items-center justify-center text-white font-bold text-lg shadow-md ${
                            index === 0 ? 'bg-gradient-to-r from-orange-500 to-orange-600' : 
                            index === 1 ? 'bg-gradient-to-r from-amber-500 to-amber-600' : 
                            'bg-gradient-to-r from-yellow-500 to-yellow-600'
                          }`}>
                            {index + 1}
                          </div>
                          <span className="font-bold text-gray-900 text-base">{pred.cuisine}</span>
                        </div>
                        <span className="text-sm font-semibold text-gray-600">{pred.confidence.toFixed(1)}%</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              <div className="mt-8 pt-6 border-t-2 border-orange-200">
                <p className="text-sm font-bold text-gray-600 mb-3">Features used for classification:</p>
                <div className="grid grid-cols-2 gap-3 text-sm">
                  <div className="bg-white/70 backdrop-blur-sm p-3 rounded-lg">
                    <span className="text-gray-500">City:</span> <span className="font-bold">{formData.city}</span>
                  </div>
                  <div className="bg-white/70 backdrop-blur-sm p-3 rounded-lg">
                    <span className="text-gray-500">Votes:</span> <span className="font-bold">{formData.votes}</span>
                  </div>
                  <div className="bg-white/70 backdrop-blur-sm p-3 rounded-lg">
                    <span className="text-gray-500">Price:</span> <span className="font-bold">{'₹'.repeat(formData.price_range)}</span>
                  </div>
                  <div className="bg-white/70 backdrop-blur-sm p-3 rounded-lg">
                    <span className="text-gray-500">Table Booking:</span> <span className="font-bold">{formData.has_table_booking ? 'Yes' : 'No'}</span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default CuisineClassification;