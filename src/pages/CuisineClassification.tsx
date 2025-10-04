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
      <div className="min-h-screen bg-gray-50 flex items-center justify-center p-4">
        <div className="bg-white shadow-xl rounded-xl p-8 w-full max-w-4xl">
          <div className="flex items-center justify-center mb-6">
            <Loader2 className="w-12 h-12 text-orange-500 animate-spin" />
          </div>
          <p className="text-center text-gray-600">Loading classification options...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 p-4">
      <div className="max-w-6xl mx-auto">
        <div className="bg-white shadow-xl rounded-xl p-8 mb-6">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-3">
              <Utensils className="w-10 h-10 text-orange-500" />
              <div>
                <h2 className="text-3xl font-bold text-gray-800">Cuisine Classification</h2>
                <p className="text-gray-500 text-sm">Multi-class classification using Random Forest</p>
              </div>
            </div>
            <button
              onClick={() => setShowPerformance(!showPerformance)}
              className="flex items-center gap-2 px-4 py-2 bg-orange-100 hover:bg-orange-200 text-orange-700 rounded-lg transition-colors"
            >
              <TrendingUp className="w-4 h-4" />
              Model Performance
            </button>
          </div>

          {showPerformance && performance && (
            <div className="mb-6 p-6 bg-orange-50 border border-orange-200 rounded-lg">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Model Performance Metrics</h3>
              
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                <div className="bg-white p-4 rounded-lg shadow-sm">
                  <div className="text-2xl font-bold text-orange-600">
                    {(performance.test_metrics.accuracy * 100).toFixed(1)}%
                  </div>
                  <div className="text-xs text-gray-600">Accuracy</div>
                </div>
                <div className="bg-white p-4 rounded-lg shadow-sm">
                  <div className="text-2xl font-bold text-blue-600">
                    {(performance.test_metrics.precision_macro * 100).toFixed(1)}%
                  </div>
                  <div className="text-xs text-gray-600">Precision</div>
                </div>
                <div className="bg-white p-4 rounded-lg shadow-sm">
                  <div className="text-2xl font-bold text-green-600">
                    {(performance.test_metrics.recall_macro * 100).toFixed(1)}%
                  </div>
                  <div className="text-xs text-gray-600">Recall</div>
                </div>
                <div className="bg-white p-4 rounded-lg shadow-sm">
                  <div className="text-2xl font-bold text-purple-600">
                    {(performance.test_metrics.f1_macro * 100).toFixed(1)}%
                  </div>
                  <div className="text-xs text-gray-600">F1-Score</div>
                </div>
              </div>

              <div className="bg-white p-4 rounded-lg">
                <h4 className="font-semibold text-gray-900 mb-3">Per-Cuisine Performance</h4>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b">
                        <th className="text-left py-2 px-2">Cuisine</th>
                        <th className="text-right py-2 px-2">Precision</th>
                        <th className="text-right py-2 px-2">Recall</th>
                        <th className="text-right py-2 px-2">F1-Score</th>
                        <th className="text-right py-2 px-2">Support</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(performance.per_cuisine_metrics).slice(0, 10).map(([cuisine, metrics]) => (
                        <tr key={cuisine} className="border-b hover:bg-gray-50">
                          <td className="py-2 px-2 font-medium">{cuisine}</td>
                          <td className="text-right py-2 px-2">{(metrics.precision * 100).toFixed(1)}%</td>
                          <td className="text-right py-2 px-2">{(metrics.recall * 100).toFixed(1)}%</td>
                          <td className="text-right py-2 px-2">
                            <span className={metrics['f1-score'] < 0.7 ? 'text-red-600 font-semibold' : ''}>
                              {(metrics['f1-score'] * 100).toFixed(1)}%
                            </span>
                          </td>
                          <td className="text-right py-2 px-2">{metrics.support}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <p className="text-xs text-gray-500 mt-3">
                  <AlertCircle className="w-3 h-3 inline mr-1" />
                  Red F1-scores indicate cuisines with classification challenges (potential biases or low support)
                </p>
              </div>
            </div>
          )}

          <div className="flex justify-center mb-6">
            <button
              onClick={loadRandomSample}
              className="flex items-center gap-2 px-4 py-2 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg transition-colors"
              disabled={loading}
            >
              <RefreshCw className="w-4 h-4" />
              Load Random Sample
            </button>
          </div>

          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  City
                </label>
                <select
                  name="city"
                  value={formData.city}
                  onChange={handleChange}
                  className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-orange-500 focus:border-transparent"
                  required
                >
                  {options?.cities.map((city, index) => (
                    <option key={index} value={city}>{city}</option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Number of Votes
                </label>
                <input
                  type="number"
                  name="votes"
                  value={formData.votes}
                  onChange={handleChange}
                  min={options?.votes_range.min || 0}
                  max={options?.votes_range.max || 10000}
                  className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-orange-500 focus:border-transparent"
                  required
                />
                <p className="text-xs text-gray-500 mt-1">
                  Range: {options?.votes_range.min} - {options?.votes_range.max}
                </p>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Price Range
                </label>
                <select
                  name="price_range"
                  value={formData.price_range}
                  onChange={handleChange}
                  className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-orange-500 focus:border-transparent"
                  required
                >
                  {options?.price_ranges && Object.entries(options.price_ranges).map(([value, label]) => (
                    <option key={value} value={value}>{label}</option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Table Booking
                </label>
                <select
                  name="has_table_booking"
                  value={formData.has_table_booking}
                  onChange={handleChange}
                  className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-orange-500 focus:border-transparent"
                >
                  <option value="1">Yes</option>
                  <option value="0">No</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Online Delivery
                </label>
                <select
                  name="has_online_delivery"
                  value={formData.has_online_delivery}
                  onChange={handleChange}
                  className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-orange-500 focus:border-transparent"
                >
                  <option value="1">Yes</option>
                  <option value="0">No</option>
                </select>
              </div>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full bg-orange-600 text-white py-3 rounded-md font-medium hover:bg-orange-700 transition-colors disabled:bg-orange-300 disabled:cursor-not-allowed flex items-center justify-center"
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
            <div className="mt-6 p-4 bg-red-50 border border-red-200 rounded-md">
              <p className="text-red-600 text-sm">{error}</p>
            </div>
          )}

          {result && (
            <div className="mt-6 p-6 bg-gradient-to-br from-orange-50 to-yellow-50 border border-orange-200 rounded-lg">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Classification Result</h3>
              
              <div className="text-center mb-6">
                <div className="text-5xl font-bold text-orange-600 mb-2">
                  {result.predicted_cuisine}
                </div>
                <div className="text-sm text-gray-600">Predicted Cuisine Type</div>
              </div>

              <div className="mb-6">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-gray-700">Confidence Level</span>
                  <span className="text-sm font-semibold text-orange-600">{result.confidence.toFixed(1)}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-3">
                  <div
                    className="bg-gradient-to-r from-orange-500 to-yellow-500 h-3 rounded-full transition-all duration-500"
                    style={{ width: `${result.confidence}%` }}
                  ></div>
                </div>
              </div>

              {result.top_predictions && result.top_predictions.length > 1 && (
                <div className="mt-6 pt-4 border-t border-orange-200">
                  <p className="text-sm font-semibold text-gray-700 mb-3">Top 3 Predictions:</p>
                  <div className="space-y-2">
                    {result.top_predictions.map((pred, index) => (
                      <div key={index} className="flex items-center justify-between bg-white p-3 rounded-lg">
                        <div className="flex items-center gap-3">
                          <div className={`w-8 h-8 rounded-full flex items-center justify-center text-white font-bold ${
                            index === 0 ? 'bg-orange-500' : index === 1 ? 'bg-orange-400' : 'bg-orange-300'
                          }`}>
                            {index + 1}
                          </div>
                          <span className="font-medium text-gray-900">{pred.cuisine}</span>
                        </div>
                        <span className="text-sm text-gray-600">{pred.confidence.toFixed(1)}%</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              <div className="mt-6 pt-4 border-t border-orange-200">
                <p className="text-sm text-gray-600 mb-2">Features used for classification:</p>
                <div className="grid grid-cols-2 gap-2 text-sm">
                  <div className="bg-white p-2 rounded">
                    <span className="text-gray-500">City:</span> <span className="font-medium">{formData.city}</span>
                  </div>
                  <div className="bg-white p-2 rounded">
                    <span className="text-gray-500">Votes:</span> <span className="font-medium">{formData.votes}</span>
                  </div>
                  <div className="bg-white p-2 rounded">
                    <span className="text-gray-500">Price:</span> <span className="font-medium">{'₹'.repeat(formData.price_range)}</span>
                  </div>
                  <div className="bg-white p-2 rounded">
                    <span className="text-gray-500">Table Booking:</span> <span className="font-medium">{formData.has_table_booking ? 'Yes' : 'No'}</span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        <div className="bg-white shadow-lg rounded-xl p-6">
          <h3 className="text-xl font-bold text-gray-900 mb-4">About This Task</h3>
          <div className="prose prose-sm max-w-none text-gray-600">
            <p className="mb-3">
              <strong>Task 3: Cuisine Classification</strong> - This multi-class classification model predicts
              restaurant cuisine types based on various features including location, price range, and service offerings.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
              <div>
                <h4 className="font-semibold text-gray-900 mb-2">Implementation Details:</h4>
                <ul className="list-disc list-inside space-y-1 text-sm">
                  <li>Algorithm: Random Forest Classifier</li>
                  <li>Features: City, Votes, Price Range, Services</li>
                  <li>Classes: Top 10 cuisines by frequency</li>
                  <li>Train/Test Split: 80/20 (stratified)</li>
                </ul>
              </div>
              <div>
                <h4 className="font-semibold text-gray-900 mb-2">Evaluation Metrics:</h4>
                <ul className="list-disc list-inside space-y-1 text-sm">
                  <li>Accuracy, Precision, Recall, F1-Score</li>
                  <li>Per-cuisine performance analysis</li>
                  <li>Confusion matrix for bias detection</li>
                  <li>Class imbalance identification</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default CuisineClassification;