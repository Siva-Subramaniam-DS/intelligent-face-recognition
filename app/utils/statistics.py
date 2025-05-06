import json
import os
from datetime import datetime
from collections import defaultdict

class RecognitionStatistics:
    def __init__(self, log_file='recognition_logs.json'):
        self.log_file = log_file
        self.logs = []
        self.stats = {
            'total_attempts': 0,
            'successful_recognitions': 0,
            'failed_recognitions': 0,
            'person_stats': defaultdict(lambda: {'attempts': 0, 'successes': 0})
        }
        self.load_logs()
    
    def load_logs(self):
        """Load existing logs from file"""
        if os.path.exists(self.log_file):
            with open(self.log_file, 'r') as f:
                self.logs = json.load(f)
                self._update_stats_from_logs()
    
    def _update_stats_from_logs(self):
        """Update statistics from loaded logs"""
        self.stats = {
            'total_attempts': len(self.logs),
            'successful_recognitions': sum(1 for log in self.logs if log['status'] == 'success'),
            'failed_recognitions': sum(1 for log in self.logs if log['status'] == 'failed'),
            'person_stats': defaultdict(lambda: {'attempts': 0, 'successes': 0})
        }
        
        for log in self.logs:
            person = log['person']
            self.stats['person_stats'][person]['attempts'] += 1
            if log['status'] == 'success':
                self.stats['person_stats'][person]['successes'] += 1
    
    def add_log(self, person, confidence, status):
        """Add a new recognition log"""
        log_entry = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'person': person,
            'confidence': float(confidence),
            'status': status
        }
        
        self.logs.append(log_entry)
        self._update_stats()
        self._save_logs()
    
    def _update_stats(self):
        """Update statistics based on the latest log"""
        latest_log = self.logs[-1]
        self.stats['total_attempts'] += 1
        
        if latest_log['status'] == 'success':
            self.stats['successful_recognitions'] += 1
        else:
            self.stats['failed_recognitions'] += 1
        
        person = latest_log['person']
        self.stats['person_stats'][person]['attempts'] += 1
        if latest_log['status'] == 'success':
            self.stats['person_stats'][person]['successes'] += 1
    
    def _save_logs(self):
        """Save logs to file"""
        with open(self.log_file, 'w') as f:
            json.dump(self.logs, f, indent=2)
    
    def get_statistics(self, page=1, per_page=20):
        """
        Get current statistics with paginated logs
        
        Args:
            page (int): The page number (1-indexed)
            per_page (int): Number of logs per page
        """
        # Reverse logs to get latest first
        reversed_logs = list(reversed(self.logs))
        
        # Calculate indices for pagination
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        
        # Get paginated logs
        paginated_logs = reversed_logs[start_idx:end_idx]
        
        # Calculate total pages
        total_pages = (len(self.logs) + per_page - 1) // per_page
        
        return {
            'total_faces': len(self.stats['person_stats']),
            'total_attempts': self.stats['total_attempts'],
            'successful_recognitions': self.stats['successful_recognitions'],
            'accuracy': round((self.stats['successful_recognitions'] / self.stats['total_attempts'] * 100) if self.stats['total_attempts'] > 0 else 0, 2),
            'recognition_logs': paginated_logs,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total_pages': total_pages,
                'total_logs': len(self.logs)
            }
        }
    
    def get_person_statistics(self, person_name):
        """Get statistics for a specific person"""
        if person_name in self.stats['person_stats']:
            stats = self.stats['person_stats'][person_name]
            return {
                'attempts': stats['attempts'],
                'successes': stats['successes'],
                'accuracy': round((stats['successes'] / stats['attempts'] * 100) if stats['attempts'] > 0 else 0, 2)
            }
        return None 