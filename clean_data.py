import pandas as pd

# 1. قراءة الملف الضخم (تأكد من كتابة المسار الصحيح للملف)
df = pd.read_csv('train_transaction.csv')

# 2. تقسيم البيانات لضمان وجود حالات احتيال في العينة
fraud = df[df['isFraud'] == 1].head(25)   # نأخذ أول 25 حالة احتيال
safe = df[df['isFraud'] == 0].head(75)    # نأخذ أول 75 حالة سليمة

# 3. دمجهم في ملف واحد (الإجمالي 100 سطر)
demo_df = pd.concat([fraud, safe])

# 4. إعادة ترتيب البيانات بشكل عشوائي لكي لا تظهر الحالات متتالية
demo_df = demo_df.sample(frac=1).reset_index(drop=True)

# 5. حفظ الملف الجديد (هذا هو الملف الذي سترفعه في الفرونت إند)
demo_df.to_csv('demo_data.csv', index=False)

print("تم إنشاء ملف العرض بنجاح: demo_data.csv")