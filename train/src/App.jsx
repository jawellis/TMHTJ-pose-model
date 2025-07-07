import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
import PoseDetection from "./components/PoseDetection";
import PoseClassifier from "./components/PoseClassifier";

function App() {
  return (
    <Router>
      <nav style={{ margin: 16 }}>
        <Link to="/pose-detection" style={{ marginRight: 12 }}>Pose Detection</Link>
        <Link to="/pose-classifier">Pose Classifier</Link>
      </nav>
      <Routes>
        <Route path="/pose-detection" element={<PoseDetection />} />
        <Route path="/pose-classifier" element={<PoseClassifier />} />
        <Route path="/" element={<PoseDetection />} />
      </Routes>
    </Router>
  );
}
export default App;
