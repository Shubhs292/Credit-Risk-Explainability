# Initialize models
models = {
    'XGBoost': XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, eval_metric='logloss'),
    'CatBoost': CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1, random_state=42, verbose=0),
    'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
}

# Train models and store results
results = {}
trained_models = {}
confusion_matrices = {}

print("Training models...")

for name, model in models.items():
    print(f"\n{name}:")
    
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    confusion_matrices[name] = cm
    
    results[name] = {
        'Accuracy': accuracy,
        'AUC': auc,
        'F1-Score': f1,
        'CV AUC Mean': cv_scores.mean(),
        'CV AUC Std': cv_scores.std()
    }
    
    trained_models[name] = model
    
    # Print Summary
    print(f"  Accuracy:     {accuracy:.4f}")
    print(f"  AUC:          {auc:.4f}")
    print(f"  F1-Score:     {f1:.4f}")
    print(f"  CV AUC:       {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    print(f"\n  Confusion Matrix: TN={cm[0,0]:,} | FP={cm[0,1]:,} | FN={cm[1,0]:,} | TP={cm[1,1]:,}")
    
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Paid', 'Default']))

print("\nAll models trained successfully.")

# Visualize Confusion Matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (name, cm) in enumerate(confusion_matrices.items()):
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=axes[idx],
                cbar_kws={'label': 'Count'},
                xticklabels=['Paid', 'Default'],
                yticklabels=['Paid', 'Default'])
    
    axes[idx].set_title(f'{name} Confusion Matrix', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Predicted', fontsize=10)
    axes[idx].set_ylabel('Actual', fontsize=10)
    
    # Add performance metrics as text
    accuracy = results[name]['Accuracy']
    auc = results[name]['AUC']
    f1 = results[name]['F1-Score']
    
    textstr = f'Accuracy: {accuracy:.3f}\nAUC: {auc:.3f}\nF1: {f1:.3f}'
    axes[idx].text(0.02, 0.98, textstr, transform=axes[idx].transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()

# Compare model performance
results_df = pd.DataFrame(results).T
print("\nModel Performance Comparison:")
print(results_df.round(4))

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

metrics = ['Accuracy', 'AUC', 'F1-Score']
for idx, metric in enumerate(metrics):
    results_df[metric].plot(kind='bar', ax=axes[idx], color= palette)
    axes[idx].set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
    axes[idx].set_ylabel(metric, fontsize=10)
    axes[idx].set_xlabel('Model', fontsize=10)
    axes[idx].set_xticklabels(results_df.index, rotation=45)
    axes[idx].grid(alpha=0.3)
    
    # Add value labels
    for container in axes[idx].containers:
        axes[idx].bar_label(container, fmt='%.3f', padding=3)

plt.tight_layout()
plt.show()

# Calculate SHAP values for all models
shap_values_dict = {}
explainers_dict = {}

# Use a subset for faster computation
X_test_sample = X_test.iloc[:500]

for name, model in trained_models.items():
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_sample)
    
    # For binary classification, some models return list of arrays
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Take positive class
    
    shap_values_dict[name] = shap_values
    explainers_dict[name] = explainer
    
    print(f"SHAP values computed for {name}")

# SHAP Summary Plot for each model
for name in trained_models.keys():
    print(f"SHAP Summary Plot: {name}")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_dict[name], X_test_sample, show=False,color= palette )
    plt.title(f'{name} SHAP Feature Importance', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()

# SHAP Bar Plot - Mean absolute SHAP values
for name in trained_models.keys():
    print(f"SHAP Bar Plot: {name}")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_dict[name], X_test_sample, plot_type='bar', show=False, color = palette)
    plt.title(f'{name} Mean |SHAP Value|', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()

# Compare top 5 features across models
top_features_comparison = {}

for name in trained_models.keys():
    mean_abs_shap = np.abs(shap_values_dict[name]).mean(axis=0)
    feature_importance = pd.DataFrame({ 'feature': final_features,'importance': mean_abs_shap }).sort_values('importance', ascending=False)
    
    top_features_comparison[name] = feature_importance.head(5)

# Display comparison
print("\nTop 5 Features by SHAP Importance:\n")
for name, features in top_features_comparison.items():
    print(f"\n{name}:")
    print(features.to_string(index=False))

# Compare top 5 features across models
top_features_comparison = {}

for name in trained_models.keys():
    mean_abs_shap = np.abs(shap_values_dict[name]).mean(axis=0)
    feature_importance = pd.DataFrame({'feature': final_features,'importance': mean_abs_shap}).sort_values('importance', ascending=False)
    
    top_features_comparison[name] = feature_importance.head(5)

# Display comparison
print("\nTop 5 Features by SHAP Importance:\n")
for name, features in top_features_comparison.items():
    print(f"\n{name}:")
    print(features.to_string(index=False))

print("FEATURE DOMINANCE ANALYSIS")

# Analyze feature dominance for XGBoost (our best model)
mean_abs_shap_xgb = np.abs(shap_values_dict['XGBoost']).mean(axis=0)
total_shap = mean_abs_shap_xgb.sum()

dominance_df = pd.DataFrame({
    'Feature': final_features,
    'SHAP_Importance': mean_abs_shap_xgb,
    'Percentage': (mean_abs_shap_xgb / total_shap) * 100
}).sort_values('SHAP_Importance', ascending=False)

print("\n📊 Feature Dominance Breakdown (XGBoost):\n")
print(dominance_df.head(10).to_string(index=False))

# Visualize dominance
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Bar chart
dominance_df.head(10).plot(kind='barh', x='Feature', y='Percentage', 
                            ax=axes[0], color=palette[3], legend=False)
axes[0].set_title('Top 10 Features: Percentage of Total SHAP Importance', 
                  fontsize=12, fontweight='bold')
axes[0].set_xlabel('Percentage of Total Importance (%)', fontsize=11)
axes[0].set_ylabel('Feature', fontsize=11)
axes[0].invert_yaxis()
axes[0].grid(alpha=0.3, axis='x')

# Add percentage labels
for idx, (feature, percentage) in enumerate(zip(dominance_df.head(10)['Feature'], 
                                                  dominance_df.head(10)['Percentage'])):
    axes[0].text(percentage + 1, idx, f'{percentage:.1f}%', 
                va='center', fontsize=9)

# Pie chart for top 5 + others
top_5 = dominance_df.head(5)
others_pct = dominance_df.iloc[5:]['Percentage'].sum()
pie_data = list(top_5['Percentage']) + [others_pct]
pie_labels = list(top_5['Feature']) + ['Others']

axes[1].pie(pie_data, labels=pie_labels, autopct='%1.1f%%', 
            colors=palette + ['#cccccc'], startangle=90)
axes[1].set_title('Feature Importance Distribution', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

# Flag dominance issues

print("DOMINANCE ASSESSMENT")

top_feature = dominance_df.iloc[0]
top_feature_pct = top_feature['Percentage']
second_feature_pct = dominance_df.iloc[1]['Percentage']

print(f"\n🔍 Top Feature: {top_feature['Feature']}")
print(f"   Importance: {top_feature_pct:.2f}% of total")
print(f"   Gap to #2: {top_feature_pct - second_feature_pct:.2f} percentage points")

if top_feature_pct > 60:
    print(f"\n🚨 CRITICAL: '{top_feature['Feature']}' accounts for {top_feature_pct:.1f}% of total importance!")
    print("   This extreme dominance suggests:")
    print("   • Potential data leakage")
    print("   • Model over-reliance on single feature")
    print("   • High fragility risk")
elif top_feature_pct > 40:
    print(f"\n WARNING: '{top_feature['Feature']}' accounts for {top_feature_pct:.1f}% of total importance")
    print("   This suggests significant feature dominance - investigate further")
else:
    print(f"\n✅ Feature distribution is relatively balanced")
    print(f"   Top feature: {top_feature_pct:.1f}% (under 40% threshold)")

# Calculate Gini coefficient for feature importance distribution
def gini_coefficient(x):
    """CalculateS Gini coefficient for inequality measurement"""
    sorted_x = np.sort(x)
    n = len(x)
    cumsum = np.cumsum(sorted_x)
    return (2 * np.sum((n - np.arange(1, n + 1) + 1) * sorted_x)) / (n * np.sum(sorted_x)) - (n + 1) / n

gini = gini_coefficient(dominance_df['SHAP_Importance'].values)
print(f"\n📈 Gini Coefficient: {gini:.3f}")
print(f"   (0 = perfect equality, 1 = maximum inequality)")

if gini > 0.7:
    print("   🚨 High inequality - feature importance very concentrated")
elif gini > 0.5:
    print("   ⚠️  Moderate inequality - some features dominate")
else:
    print("   ✅ Low inequality - relatively balanced feature usage")

# SHAP Dependence Plots for top features (using XGBoost as example)
model_name = 'XGBoost'
top_features = ['int_rate', 'out_prncp', 'fico_score', 'annual_inc']

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.ravel()

for idx, feature in enumerate(top_features):
    shap.dependence_plot(feature, shap_values_dict[model_name], X_test_sample, 
                         ax=axes[idx], show=False)
    axes[idx].set_title(f'SHAP Dependence: {feature}', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

