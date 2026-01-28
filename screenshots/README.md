# Screenshot Instructions

## Required Screenshots (4 total)

You need to take these screenshots from your Kaggle notebook runs:

### 1. training_progress.png
**What to capture:**
- The training cell output showing epochs completing
- Training loss decreasing
- Final "Model saved" message

**Screenshot the output like:**
```
Epoch 1/3: 100% loss: 0.xxx
Epoch 2/3: 100% loss: 0.xxx
Epoch 3/3: 100% loss: 0.xxx
Model saved to models/lora-adapter/
```

---

### 2. base_vs_finetuned.png
**What to capture:**
- The comparison cell output showing both models
- The "Base Model" section header
- The "Fine-Tuned Model" section header
- The analysis at the bottom showing tag detection

**Screenshot the output like:**
```
======================================================================
                    BASE MODEL (Prompt-Only)
======================================================================
<analysis>
User wants to create a new task...
</analysis>
...

======================================================================
                  FINE-TUNED MODEL (LoRA)
======================================================================
<analysis>
User wants to create a new task...
</analysis>
...

Analysis:
Base Model - Has <action> tag: True/False
Fine-Tuned - Has <action> tag: True
```

---

### 3. tool_execution_add.png
**What to capture:**
- The add_task test cell output
- Shows user query
- Shows model response with <analysis> and <action> tags
- Shows EXECUTION RESULT with success message

**Screenshot the output like:**
```
USER: Add a task to buy groceries tomorrow
============================================================
AGENT RESPONSE:
<analysis>
User wants to create a new task...
</analysis>

<action>
add_task(title="Buy groceries", due_date="2026-01-26")
</action>

EXECUTION RESULT:
{'status': 'success', 'message': "Task added: 'Buy groceries'", 'task_count': 1}
```

---

### 4. conversation_response.png
**What to capture:**
- The "Hello! How are you?" test cell
- Shows model using <final> tag instead of <action>
- Shows DIRECT RESPONSE

**Screenshot the output like:**
```
USER: Hello! How are you?
============================================================
AGENT RESPONSE:
<final>
Hello! I'm good, thanks for asking. How can I help you manage your tasks today?
</final>

DIRECT RESPONSE: Hello! I'm good, thanks for asking...
```

---

## How to Take Screenshots

### On Windows:
1. Use **Win + Shift + S** to open Snipping Tool
2. Select the area to capture
3. Save as PNG

### On Mac:
1. Use **Cmd + Shift + 4**
2. Select area
3. Find screenshot on Desktop

### On Kaggle:
1. Run the notebook cells
2. Scroll to the output you want
3. Take screenshot of that section

---

## File Naming

Save screenshots with EXACT names:
```
screenshots/
├── training_progress.png
├── base_vs_finetuned.png
├── tool_execution_add.png
└── conversation_response.png
```

---

## After Adding Screenshots

1. Commit and push to GitHub:
```bash
git add screenshots/
git commit -m "Add project screenshots"
git push
```

2. Screenshots will display in README automatically
