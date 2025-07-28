# FIFA-2026-Toronto-Crowd-Optimization-
A multi-method operations research project developed as part of MMAI 861 – Analytical Decision Making at Queen’s University (Smith School of Business). This model addresses the challenge of evacuating 100,000 fans from BMO Field in Toronto during the FIFA World Cup 2026 — under real-world constraints, infrastructure limits, and budget restrictions.


🔍 Objective
Design a time- and cost-efficient post-event evacuation plan for FIFA 2026 using a blend of optimization tools and simulation techniques.

📊 Methods Used
Excel Solver – Core linear programming model to optimize vehicle deployment and signal timing.

Python Simulation – Queue-based simulation to stress-test evacuation timelines under realistic congestion.

@RISK Monte Carlo – Probabilistic modeling with variations in weather, fleet availability, attendance, and transit preferences.

PrecisionTree – Decision support model to evaluate trade-offs between cost, time, and resource use.

✅ Key Outcomes
Evacuation Time Achieved: 57.5 minutes (compared to 75–95 mins at global stadiums like Wembley or Camp Nou)

Budget Utilization: $56,000 out of a $200,000 cap

Real-World Scenarios: Weather delays, variable crowd size, fleet disruptions, transit behavioral shifts

🛠️ Technologies
Excel Solver

Python (Pandas, SimPy)

Palisade DecisionTools Suite (@RISK, PrecisionTree)

📂 Contents
/models/ – Excel Solver sheets, simulation outputs

/code/ – Python simulation scripts

/report/ – Final technical report + presentation deck

/scenarios/ – Monte Carlo inputs and results

If you're working in transit planning, emergency preparedness, or major event logistics, this repo shows how integrated modeling can inform real-world infrastructure decisions.

Pull requests welcome.
