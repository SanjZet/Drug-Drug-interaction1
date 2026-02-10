# 🚀 QUICK START - Launch in 30 Seconds!

## For Immediate Demo/Presentation:

### Method 1: One Command Launch (Recommended)
```bash
python start_server.py
```
**That's it!** Browser opens automatically at http://localhost:5000

---

### Method 2: Manual Launch
```bash
# Start server
python api_server.py

# Open browser to: http://localhost:5000
```

---

## 🎯 Demo Flow (2 minutes)

### 1. Open Website
- Launches at http://localhost:5000
- See statistics: **19,831 drugs**, **1.45M interactions**, **92.2% accuracy**

### 2. Search Drugs
Type in search box:
```
"Aspirin" → Click to select
"Warfarin" → Click to select
```

### 3. Check Interactions
- Click blue **"Check Interactions"** button
- See result: **🔴 HIGH RISK (92% probability)**
- Shows description from database

### 4. Try More Drugs
Add 2-3 more drugs:
```
"Insulin"
"Leuprolide"
"Etanercept"
```
Click **"Check Interactions"** again → Shows all pairs!

---

## 📊 Show Visualizations

Open these images from `data/` folder:

1. **network_statistics.png** - Complete metrics
2. **degree_distribution.png** - Network analysis
3. **network_spring.png** - Interactive graph
4. **training_curves.png** - Model performance

---

## 🔧 If Something Goes Wrong

### Port Already in Use:
```bash
# Windows: Kill existing Python
taskkill /F /IM python.exe

# Mac/Linux: Find and kill process
lsof -ti:5000 | xargs kill -9

# Try again
python start_server.py
```

### Missing Dependencies:
```bash
pip install flask flask-cors torch pandas numpy
python start_server.py
```

### Data Files Missing:
```bash
python run_pipeline.py
# Choose: 1 (Parse), 2 (Build Graph), 3 (Train Model)
python start_server.py
```

---

## 🎬 Presentation Script

**Slide 1:** "Today I'll demo a Drug Interaction Checker using AI"

**Slide 2:** *Launch web interface*  
"Here's our system - analyzing 19,831 drugs and 1.45 million interactions"

**Slide 3:** *Type "Aspirin"*  
"Real-time search through DrugBank pharmaceutical database"

**Slide 4:** *Select "Aspirin" and "Warfarin"*  
"Multiple drug selection for comprehensive checking"

**Slide 5:** *Click "Check Interactions"*  
"System checks known database and uses AI prediction - result: HIGH RISK!"

**Slide 6:** *Show network visualization images*  
"Our Graph Convolutional Network learned patterns from this drug interaction network"

**Slide 7:** *Add more drugs, show batch results*  
"Can check multiple drugs simultaneously - finds all interaction pairs"

**Slide 8:** "Model achieved 92.2% accuracy - ready for real-world use!"

---

## 📱 Access Web Interface

Once running, these URLs work:

- **Local**: http://localhost:5000
- **Same network**: http://YOUR_IP:5000
- **API docs**: http://localhost:5000/api/health

---

## 🎉 You're Ready!

Everything is set up and working. Just run:

```bash
python start_server.py
```

**Good luck with your presentation!** 🎓
