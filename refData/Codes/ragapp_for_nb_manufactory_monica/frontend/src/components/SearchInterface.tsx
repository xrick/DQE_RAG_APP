// frontend/src/components/SearchInterface.tsx
import React, { useState } from 'react';
import { SearchResults } from './SearchResults';
import { searchAPI } from '../services/api';

export const SearchInterface: React.FC = () => {
    const [query, setQuery] = useState('');
    const [loading, setLoading] = useState(false);
    const [results, setResults] = useState<SearchResults | null>(null);
    
    const handleSearch = async () => {
        try {
            setLoading(true);
            const data = await searchAPI.search(query);
            setResults(data);
        } catch (error) {
            console.error('Search failed:', error);
        } finally {
            setLoading(false);
        }
    };
    
    return (
        <div className="container mx-auto p-4">
            <div className="flex gap-4 mb-4">
                <input
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    className="flex-1 p-2 border rounded"
                    placeholder="輸入您的問題..."
                />
                <button
                    onClick={handleSearch}
                    disabled={loading}
                    className={`px-4 py-2 rounded ${
                        loading ? 'bg-gray-400' : 'bg-blue-500 hover:bg-blue-600'
                    } text-white`}
                >
                    {loading ? '搜索中...' : '搜索'}
                </button>
            </div>
            
            {results && <SearchResults results={results} />}
        </div>
    );
};
