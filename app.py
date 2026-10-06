import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load Dataset
df = pd.read_csv("careers.csv")

# Page Settings
st.set_page_config(
    page_title="AI Career Recommendation System",
    page_icon="🎓",
    layout="wide"
)

# Title
st.title("🎓 AI Career Recommendation System")
st.write("Find the best career path based on your skills and interests.")

st.markdown("---")

# Student Details
name = st.text_input("👤 Student Name")

education = st.selectbox(
    "🎓 Education",
    ["B.Tech", "BCA", "B.Sc", "MCA", "M.Tech"]
)

year = st.selectbox(
    "📚 Year",
    ["1st Year", "2nd Year", "3rd Year", "4th Year"]
)

skills = st.text_area(
    "💻 Enter Your Skills",
    placeholder="Example: python machine learning html css"
)

# Button
if st.button("🚀 Recommend Career"):

    if skills.strip() == "":
        st.warning("Please enter your skills.")
    else:

        # Convert skills to vectors
        vectorizer = CountVectorizer()

        career_vectors = vectorizer.fit_transform(df["Skills"])

        user_vector = vectorizer.transform([skills])

        similarity = cosine_similarity(
            user_vector,
            career_vectors
        )[0]

        # Best Career
        best_index = similarity.argmax()

        career = df.iloc[best_index]

        st.success(
            f"✅ Recommended Career: {career['Career']}"
        )

        st.markdown("---")

        # Description
        st.subheader("📖 Career Description")
        st.write(career["Description"])

        # Salary
        st.subheader("💰 Average Salary")
        st.write(career["Salary"])

        st.markdown("---")

        # Top 3 Careers
        st.subheader("🏆 Top 3 Career Matches")

        top3 = similarity.argsort()[-3:][::-1]

        career_scores = {}

        for i in top3:
            score = round(similarity[i] * 100, 2)

            st.write(
                f"**{df.iloc[i]['Career']}** : {score}%"
            )

            career_scores[df.iloc[i]["Career"]] = score

        st.markdown("---")

        # Match Percentage Table
        st.subheader("📊 Career Match Percentage")

        result_df = pd.DataFrame({
            "Career": df["Career"],
            "Match %": [
                round(score * 100, 2)
                for score in similarity
            ]
        })

        st.dataframe(result_df)

        st.bar_chart(
            result_df.set_index("Career")
        )

        st.markdown("---")

        # Missing Skills
        st.subheader("❌ Missing Skills")

        user_skills = set(
            skills.lower()
            .replace(",", " ")
            .split()
        )

        career_skills = set(
            career["Skills"]
            .replace(",", " ")
            .split()
        )

        missing = career_skills - user_skills

        if len(missing) == 0:
            st.success(
                "You already have all required skills!"
            )
        else:
            for skill in missing:
                st.write(f"❌ {skill}")

        st.markdown("---")

        # Learning Roadmap
        st.subheader("🛣 Learning Roadmap")

        if len(missing) == 0:
            st.write(
                "Start building projects and internships."
            )
        else:
            month = 1

            for skill in missing:
                st.write(
                    f"Month {month}: Learn {skill}"
                )
                month += 1

        st.markdown("---")

        # Summary
        st.subheader("📋 Student Summary")

        st.write(f"👤 Name: {name}")
        st.write(f"🎓 Education: {education}")
        st.write(f"📚 Year: {year}")
        st.write(
            f"🚀 Best Career Choice: {career['Career']}"
        )

        st.success(
            "Career Analysis Completed Successfully!"
        )