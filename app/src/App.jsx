import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Home from "./pages/Home";
import StepTrainer from "./pages/StepTrainer";
import FinalStep from "./pages/FinalStep";

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/train" element={<StepTrainer />} />
        <Route path="/final" element={<FinalStep />} />
      </Routes>
    </Router>
  );
}

export default App;
