from graphviz import Digraph

dot = Digraph(comment="ML Workflow", format="png")
dot.attr(rankdir="LR", size="8")

dot.node("raw", "Raw Data (UCI CSV)", shape="box", style="filled", fillcolor="lightblue")
dot.node("pre", "Preprocessing\n(clean + encode)", shape="box", style="filled", fillcolor="lightyellow")
dot.node("train", "Model Training\n(LogReg, RF, XGBoost)", shape="box", style="filled", fillcolor="lightgreen")
dot.node("eval", "Evaluation & Selection", shape="box", style="filled", fillcolor="lightpink")
dot.node("model", "Best Model\n(final_model.pkl)", shape="cylinder", style="filled", fillcolor="orange")
dot.node("ui", "Streamlit App\n(UI for manual & CSV input)", shape="component", style="filled", fillcolor="lightgray")

dot.edges([("raw", "pre"), ("pre", "train"), ("train", "eval"), ("eval", "model"), ("model", "ui")])

dot.render("workflow_diagram", cleanup=True)
print("✅ Saved workflow_diagram.png")
