print("DATA LEAKAGE DETECTION")

# Features that shouldn't be available at loan application time
leakage_keywords = [
    'pymnt', 'prncp', 'rec_', 'collection', 'recovery', 
    'last_', 'total_pymnt', 'out_prncp', 'next_pymnt'
]

# Identify leakage features
leakage_features = []
clean_features = []

print("\n🔍 Analyzing features for data leakage...\n")

for feature in final_features:
    is_leakage = any(keyword in feature.lower() for keyword in leakage_keywords)
    
    if is_leakage:
        leakage_features.append(feature)
        print(f" LEAKAGE: {feature}")
    else:
        clean_features.append(feature)

print(f"\n📊 Analysis Summary:")
print(f"  Total features: {len(final_features)}")
print(f"  Clean features: {len(clean_features)}")
print(f"  Leakage features: {len(leakage_features)}")
print(f"\n Clean features (available at application time):")
for feat in clean_features:
    print(f"  • {feat}")

# TRAIN CLEAN MODEL (NO LEAKAGE)

print("TRAINING CLEAN MODEL (NO DATA LEAKAGE)")

X_train_clean = X_train[clean_features]
X_test_clean = X_test[clean_features]

print(f"\n Clean dataset shape:")
print(f"  Training: {X_train_clean.shape}")
print(f"  Testing: {X_test_clean.shape}")

# Train XGBoost on clean features
print(f"\n🔧 Training XGBoost with clean features only...")
model_clean = XGBClassifier( n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, eval_metric='logloss')
model_clean.fit(X_train_clean, y_train)

# Evaluate clean model
y_pred_clean = model_clean.predict(X_test_clean)
y_pred_proba_clean = model_clean.predict_proba(X_test_clean)[:, 1]

accuracy_clean = accuracy_score(y_test, y_pred_clean)
auc_clean = roc_auc_score(y_test, y_pred_proba_clean)
f1_clean = f1_score(y_test, y_pred_clean)

# Cross-validation
cv_scores_clean = cross_val_score(model_clean, X_train_clean, y_train, cv=5, scoring='roc_auc')

print(f"\n✅ Clean Model Performance:")
print(f"  Accuracy: {accuracy_clean:.4f}")
print(f"  AUC: {auc_clean:.4f}")
print(f"  F1-Score: {f1_clean:.4f}")
print(f"  CV AUC: {cv_scores_clean.mean():.4f} (±{cv_scores_clean.std():.4f})")

# Compare with original model
print("LEAKAGE IMPACT ASSESSMENT")

original_results = results['XGBoost']

comparison_df = pd.DataFrame({
    'Metric': ['Accuracy', 'AUC', 'F1-Score'],
    'With Leakage': [original_results['Accuracy'], original_results['AUC'], original_results['F1-Score']],
    'Clean Model': [accuracy_clean, auc_clean, f1_clean],
    'Drop': [
        original_results['Accuracy'] - accuracy_clean,
        original_results['AUC'] - auc_clean,
        original_results['F1-Score'] - f1_clean
    ],
    'Drop %': [
        ((original_results['Accuracy'] - accuracy_clean) / original_results['Accuracy'] * 100),
        ((original_results['AUC'] - auc_clean) / original_results['AUC'] * 100),
        ((original_results['F1-Score'] - f1_clean) / original_results['F1-Score'] * 100)
    ]
})

print(f"\n{comparison_df.to_string(index=False)}")

# Visualize comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Bar comparison
x = np.arange(len(comparison_df['Metric']))
width = 0.35

axes[0].bar(x - width/2, comparison_df['With Leakage'], width, 
           label='With Leakage', color=palette[0])
axes[0].bar(x + width/2, comparison_df['Clean Model'], width, 
           label='Clean Model', color=palette[2])

axes[0].set_xlabel('Metric', fontsize=12)
axes[0].set_ylabel('Score', fontsize=12)
axes[0].set_title('Model Performance: With Leakage vs Clean', fontsize=14, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(comparison_df['Metric'])
axes[0].legend()
axes[0].grid(alpha=0.3, axis='y')

# Add value labels
for i, (leak, clean) in enumerate(zip(comparison_df['With Leakage'], comparison_df['Clean Model'])):
    axes[0].text(i - width/2, leak + 0.01, f'{leak:.3f}', ha='center', fontsize=9)
    axes[0].text(i + width/2, clean + 0.01, f'{clean:.3f}', ha='center', fontsize=9)

# Performance drop visualization
axes[1].bar(comparison_df['Metric'], comparison_df['Drop %'], color=palette[3])
axes[1].set_xlabel('Metric', fontsize=12)
axes[1].set_ylabel('Performance Drop (%)', fontsize=12)
axes[1].set_title('Impact of Removing Leakage Features', fontsize=14, fontweight='bold')
axes[1].grid(alpha=0.3, axis='y')
axes[1].axhline(y=10, color='orange', linestyle='--', alpha=0.7, label='10% threshold')
axes[1].axhline(y=20, color='red', linestyle='--', alpha=0.7, label='20% threshold')
axes[1].legend()

# Add value labels
for i, (metric, drop) in enumerate(zip(comparison_df['Metric'], comparison_df['Drop %'])):
    axes[1].text(i, drop + 0.5, f'{drop:.1f}%', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()

# Assessment
auc_drop_pct = ((original_results['AUC'] - auc_clean) / original_results['AUC'] * 100)

print("ASSESSMENT")

print(f"\n💡 Key Findings:")
print(f"  • Original AUC (with leakage): {original_results['AUC']:.4f}")
print(f"  • Clean AUC (no leakage): {auc_clean:.4f}")
print(f"  • Performance drop: {auc_drop_pct:.2f}%")

if auc_drop_pct > 20:
    print(f"\n SEVERE LEAKAGE IMPACT: {auc_drop_pct:.1f}% AUC drop")
    print("  The original model was heavily dependent on post-outcome information!")
    print("  Clean model represents TRUE predictive power at application time.")
elif auc_drop_pct > 10:
    print(f"\n SIGNIFICANT LEAKAGE: {auc_drop_pct:.1f}% AUC drop")
    print("  Leakage features inflated performance considerably.")
else:
    print(f"\n MINIMAL LEAKAGE: {auc_drop_pct:.1f}% AUC drop")
    print("  Model performance is relatively authentic.")

print(f"\n📌 Conclusion:")
print(f"  The clean model's {auc_clean:.4f} AUC represents what we can ACTUALLY")
print(f"  expect in production when making real loan decisions.")

# Identify top feature for XGBoost
mean_abs_shap_xgb = np.abs(shap_values_dict['XGBoost']).mean(axis=0)
top_feature_idx = np.argmax(mean_abs_shap_xgb)
top_feature_name = final_features[top_feature_idx]

print(f"Top SHAP feature for XGBoost: {top_feature_name}")
print(f"Mean |SHAP value|: {mean_abs_shap_xgb[top_feature_idx]:.4f}")

# Train model without top feature
features_without_top = [f for f in final_features if f != top_feature_name]

X_train_reduced = X_train[features_without_top]
X_test_reduced = X_test[features_without_top]

# Train new model
print(f"\nTraining XGBoost WITHOUT '{top_feature_name}'...\n")
model_reduced = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, eval_metric='logloss')
model_reduced.fit(X_train_reduced, y_train)

# Evaluate
y_pred_reduced = model_reduced.predict(X_test_reduced)
y_pred_proba_reduced = model_reduced.predict_proba(X_test_reduced)[:, 1]

accuracy_reduced = accuracy_score(y_test, y_pred_reduced)
auc_reduced = roc_auc_score(y_test, y_pred_proba_reduced)
f1_reduced = f1_score(y_test, y_pred_reduced)

# Compare
print("\nFragility Test Results:")
print(f"{'Metric':<20} {'Original':<15} {'Without Top Feature':<20} {'Drop':<10}")
 

original_results = results['XGBoost']
print(f"{'Accuracy':<20} {original_results['Accuracy']:<15.4f} {accuracy_reduced:<20.4f} {(original_results['Accuracy'] - accuracy_reduced):<10.4f}")
print(f"{'AUC':<20} {original_results['AUC']:<15.4f} {auc_reduced:<20.4f} {(original_results['AUC'] - auc_reduced):<10.4f}")
print(f"{'F1-Score':<20} {original_results['F1-Score']:<15.4f} {f1_reduced:<20.4f} {(original_results['F1-Score'] - f1_reduced):<10.4f}")
 

# Interpretation
auc_drop_pct = (original_results['AUC'] - auc_reduced) / original_results['AUC'] * 100
print(f"\nAUC dropped by {auc_drop_pct:.2f}% when removing '{top_feature_name}'")

if auc_drop_pct > 10:
    print(f"WARNING: Significant performance drop indicates OVER-RELIANCE on '{top_feature_name}'")
elif auc_drop_pct > 5:
    print(f"CAUTION: Moderate dependence on '{top_feature_name}' detected")
else:
    print(f"Model shows resilience - balanced feature contribution")

# Visualize fragility test results
comparison_data = pd.DataFrame({
    'Original Model': [original_results['Accuracy'], original_results['AUC'], original_results['F1-Score']],
    'Without Top Feature': [accuracy_reduced, auc_reduced, f1_reduced]}, index=['Accuracy', 'AUC', 'F1-Score'])

ax = comparison_data.plot(kind='bar', figsize=(10, 6), color=palette, width=0.7)
plt.title(f'Fragility Test: Performance With vs Without "{top_feature_name}"', 
          fontsize=14, fontweight='bold', pad=20)
plt.ylabel('Score', fontsize=12)
plt.xlabel('Metric', fontsize=12)
plt.xticks(rotation=0)
plt.legend(title='Model Version', fontsize=10)
plt.grid(alpha=0.3, axis='y')

# Add value labels
for container in ax.containers:
    ax.bar_label(container, fmt='%.3f', padding=3)

plt.tight_layout()
plt.show()

print("\nFragility Test Analysis:")
print(f"  The model's 94.6% AUC collapsed to 72.6% when we removed last_pymnt_amnt")
print(f"  F1-Score dropped from 0.72 to 0.24 (67% decrease)")
print(f"\nInterpretation:")
print(f"  The model learned nothing about genuine credit risk")
print(f"  It simply memorized: 'low last payment = default'")
print(f"  Performance metrics alone would not catch this")
print(f"\nConclusion: This validates that accuracy without understanding is dangerous")

print("INDIVIDUAL PREDICTION EXPLANATIONS")

# Select interesting cases to explain
# 1. Highrisk prediction (default)
# 2. Low risk prediction (paid)
# 3. Borderline case

# Get predictions for test sample
y_pred_proba_sample = trained_models['XGBoost'].predict_proba(X_test_sample)[:, 1]

# Find interesting cases
high_risk_idx = np.argmax(y_pred_proba_sample)  # Highest default probability
low_risk_idx = np.argmin(y_pred_proba_sample)    # Lowest default probability  
borderline_idx = np.argmin(np.abs(y_pred_proba_sample - 0.5))  # Closest to 50%

interesting_cases = [
    (high_risk_idx, "HIGH RISK", y_pred_proba_sample[high_risk_idx]),
    (borderline_idx, "BORDERLINE", y_pred_proba_sample[borderline_idx]),
    (low_risk_idx, "LOW RISK", y_pred_proba_sample[low_risk_idx])
]

print(f"\n📋 Selected Cases for Analysis:\n")
for idx, label, prob in interesting_cases:
    actual_status = "DEFAULT" if y_test.iloc[idx] == 1 else "PAID"
    print(f"  {label:12s} - Predicted: {prob:.1%} | Actual: {actual_status}")

# Generate force plots for each case
for case_idx, (idx, label, prob) in enumerate(interesting_cases, 1):
    print(f"CASE {case_idx}: {label} PREDICTION")
    
    # Get instance details
    instance = X_test_sample.iloc[idx]
    actual_label = "DEFAULT (1)" if y_test.iloc[idx] == 1 else "PAID (0)"
    
    print(f"\n📊 Prediction: {prob:.1%} default probability")
    print(f"🎯 Actual outcome: {actual_label}")
    print(f"\n📝 Loan Application Details:")
    
    # Show key features
    key_features_to_show = ['loan_amnt', 'int_rate', 'annual_inc', 'dti', 'fico_score']
    for feat in key_features_to_show:
        if feat in instance.index:
            print(f"  • {feat:15s}: {instance[feat]:.2f}")
    
    # Create matplotlib force plot
    print(f"\n🔍 SHAP Force Plot Breakdown:")
    
    try:
        shap.force_plot(
            explainers_dict['XGBoost'].expected_value,
            shap_values_dict['XGBoost'][idx],
            X_test_sample.iloc[idx],
            matplotlib=True,
            show=False
        )
        plt.title(f'SHAP Force Plot - {label} Case (Prediction: {prob:.1%})', 
                 fontsize=12, fontweight='bold', pad=15)
        plt.tight_layout()
        plt.show()
    except:
        # Fallback to waterfall plot if force plot fails
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values_dict['XGBoost'][idx],
                base_values=explainers_dict['XGBoost'].expected_value,
                data=X_test_sample.iloc[idx],
                feature_names=final_features
            ),
            max_display=10,
            show=True
        )
    
    # Show top contributing features
    feature_contributions = pd.DataFrame({
        'Feature': final_features,
        'Value': instance.values,
        'SHAP_Impact': shap_values_dict['XGBoost'][idx]
    }).sort_values('SHAP_Impact', key=abs, ascending=False)
    
    print(f"\n📈 Top 5 Features Pushing Prediction:")
    print(feature_contributions.head(5).to_string(index=False))

# Bias Detection: Feature Group Patterns Using SHAP + Predictions
model_name = 'XGBoost'

# Combine SHAP values with model predictions
shap_col_names = [f'{col}_shap' for col in final_features]
shap_df = pd.DataFrame(shap_values_dict[model_name], columns=shap_col_names)
shap_df['prediction'] = trained_models[model_name].predict_proba(X_test_sample)[:, 1]

# Merge features with predictions into a single DataFrame
analysis_df = X_test_sample.reset_index(drop=True).copy()
analysis_df['prediction'] = shap_df['prediction'].values

print("Bias Detection Analysis\n")

# Group analysis by home ownership
print("Average predictions by Home Ownership:")
home_shap_analysis = (analysis_df.groupby('home_ownership_encoded').agg({'int_rate': 'mean', 'fico_score': 'mean', 'prediction': 'mean'}).reset_index().round(4))
print(home_shap_analysis)
print("\nDo prediction differences reflect actual financial indicators?\n")

# Group analysis by loan purpose
print("Average predictions by Loan Purpose:")
purpose_analysis = (
    analysis_df.groupby('purpose_encoded')
    .agg({'int_rate': 'mean', 'fico_score': 'mean', 'prediction': 'mean'})
    .reset_index()
    .round(4)
)
print(purpose_analysis.head(10))
print("\nDo these prediction patterns align with risk profiles?")

# Visualize potential bias patterns 
categorical_encoded = [col for col in final_features if '_encoded' in col]

if len(categorical_encoded) >= 2:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # Plot first categorical feature
    if categorical_encoded[0] in analysis_df.columns:
        try:
            # Get column and ensure it's 1-dimensional
            cat1_col = analysis_df[categorical_encoded[0]]
            if isinstance(cat1_col, pd.DataFrame):
                cat1_col = cat1_col.iloc[:, 0]
            
            cat1_analysis = analysis_df.groupby(cat1_col)['prediction'].mean()
            cat1_analysis.plot(kind='bar', ax=axes[0], color= palette[0])
            axes[0].set_title(f'Mean Default Probability by {categorical_encoded[0]}', 
                             fontsize=12, fontweight='bold')
            axes[0].set_xlabel(f'{categorical_encoded[0]}', fontsize=10)
            axes[0].set_ylabel('Mean Predicted Default Probability', fontsize=10)
            axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)
            axes[0].grid(alpha=0.3, axis='y')
        except Exception as e:
            axes[0].text(0.5, 0.5, f'Error: {str(e)[:50]}', 
                        ha='center', va='center', transform=axes[0].transAxes)
    
    # Plot second categorical feature
    if categorical_encoded[1] in analysis_df.columns:
        try:
            # Get column and ensure it's 1-dimensional
            cat2_col = analysis_df[categorical_encoded[1]]
            if isinstance(cat2_col, pd.DataFrame):
                cat2_col = cat2_col.iloc[:, 0]
            
            cat2_analysis = analysis_df.groupby(cat2_col)['prediction'].mean()
            cat2_analysis.plot(kind='bar', ax=axes[1], color= palette[-1])
            axes[1].set_title(f'Mean Default Probability by {categorical_encoded[1]}', 
                             fontsize=12, fontweight='bold')
            axes[1].set_xlabel(f'{categorical_encoded[1]}', fontsize=10)
            axes[1].set_ylabel('Mean Predicted Default Probability', fontsize=10)
            axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)
            axes[1].grid(alpha=0.3, axis='y')
        except Exception as e:
            axes[1].text(0.5, 0.5, f'Error: {str(e)[:50]}', 
                        ha='center', va='center', transform=axes[1].transAxes)
    
    plt.tight_layout()
    plt.show()
    
    print("\nAlways verify: Are these differences justified by legitimate risk factors?")
    
elif len(categorical_encoded) == 1:
    # Plot single categorical feature
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    # Get column and ensure it's 1-dimensional
    cat_col = analysis_df[categorical_encoded[0]]
    if isinstance(cat_col, pd.DataFrame):
        cat_col = cat_col.iloc[:, 0]
                            
    cat_analysis = analysis_df.groupby(cat_col)['prediction'].mean()
    cat_analysis.plot(kind='bar', ax=ax, color= palette[0])
    ax.set_title(f'Mean Default Probability by {categorical_encoded[0]}',fontsize=12, fontweight='bold')
    ax.set_xlabel(f'{categorical_encoded[0]}', fontsize=10)
    ax.set_ylabel('Mean Predicted Default Probability', fontsize=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()
                            
    print("\nAlways verify: Are these differences justified by legitimate risk factors?")

