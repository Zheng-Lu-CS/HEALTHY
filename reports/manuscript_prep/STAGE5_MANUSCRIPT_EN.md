# Deep Phenotyping of Intrinsic Capacity in Older Adults Using Continuous-Domain Scoring and Group-Aware Masked Representation Learning

## Abstract

**Background and Objectives**  
Cross-sectional geriatric assessment datasets are increasingly used to study intrinsic capacity (IC), but many existing analyses still rely on binary domain impairment counts and low-dimensional summaries. We aimed to build a clinically grounded and methodologically transparent pipeline for deep phenotyping of IC in older adults using continuous-domain IC approximation, representation learning, and unsupervised phenotyping without longitudinal outcomes or future event labels.

**Research Design and Methods**  
We analyzed a multidimensional assessment export including 6,025 older adults aged 60-100 years and 361 raw variables. Explicit missing codes were normalized, ultra-sparse variables were excluded from the primary modeling matrix, and external functional validators were withheld from clustering. Five continuous IC domain scores (cognition, psychological, vitality, locomotion, and sensory; each scaled 0-100) were constructed as clinically informed approximations. A primary design matrix was built from 59 core structured variables plus continuous IC domains (175 postprocessed features grouped into nine clinical blocks). We compared PCA, a group-aware masked tabular encoder (GroupMaskedFT), and text-table dual-tower ablations across KMeans and Gaussian mixture models with k = 3-6. Candidate solutions were evaluated using latent geometry, bootstrap stability, cluster balance, and separation across held-out functional correlates.

**Results**  
Participants had a mean age of 71.1 years (SD 6.9); 46.7% were men. Mean continuous IC was 74.5/100, with locomotion (62.3) and sensory function (63.4) showing the lowest domain means. Among the best method-specific solutions, GroupMaskedFT with KMeans at k = 3 showed substantially clearer latent geometry than PCA with KMeans at k = 3 (silhouette 0.311 vs 0.110), with similar stability (0.988 vs 0.991) and cluster balance (minimum cluster ratio 0.240 vs 0.223), while PCA showed slightly higher external separation (0.225 vs 0.217). We selected the GroupMaskedFT 3-cluster solution as the primary manuscript model because it provided the clearest latent structure without demographic over-fragmentation. The three phenotypes represented a low-capacity phenotype (n = 1,571; overall IC 61.4), an intermediate phenotype with persistent locomotion-sensory vulnerability (n = 1,445; overall IC 73.1), and a high-capacity phenotype (n = 3,009; overall IC 82.0). These phenotypes showed graded differences in ADL total (83.3, 95.7, and 99.2), IADL dependence score (13.6, 9.7, and 8.2), sarcopenia score (3.28, 1.22, and 0.38), Fried frailty score (2.00, 1.24, and 0.72), frailty screen score (1.47, 0.53, and 0.21), fall risk score (36.8, 25.3, and 17.2), and resilience score (39.6, 44.8, and 50.3; all p < 0.001). The strongest differentiating markers were cognition total, self-rated health, locomotion total, psychological total, gait abnormality, vitality total, and age. Dual-tower semantic enhancement did not improve phenotype quality. A higher-granularity 4-cluster sensitivity solution slightly increased external separation but split the high-capacity group almost entirely by sex and was therefore treated as supplementary only.

**Discussion and Implications**  
A continuous-domain IC approximation combined with group-aware masked representation learning can recover clinically meaningful heterogeneity from cross-sectional geriatric assessment data. The deep tabular model improved latent structure, but not universal clinical superiority, over a strong PCA baseline. These findings support cautious, evidence-bound deep phenotyping of IC in older adults while underscoring the need to avoid causal or prognostic claims in purely cross-sectional settings.

**Keywords:** intrinsic capacity; older adults; phenotyping; clustering; representation learning; geriatric assessment

## Introduction

Intrinsic capacity has become a central concept in healthy aging because it shifts attention from single diseases toward the composite physical and mental capacities that support function in later life. In clinical practice and population aging research, however, IC is often operationalized in a simplified way: domains are dichotomized into impaired versus unimpaired, impairment counts are summed, and downstream analyses focus on broad risk stratification rather than heterogeneity within functional states.

This simplification is understandable but costly. Binary IC scoring discards within-domain severity, compresses clinically diverse individuals into the same category, and makes it difficult to distinguish older adults with similar overall IC counts but different patterns of vulnerability. The problem is especially relevant for multidimensional geriatric assessment data, where cognition, mood, nutrition, mobility, sensory function, multimorbidity, and self-rated health are all measured in parallel and may contain richer latent structure than a binary score can preserve.

At the same time, the current dataset reflects a common real-world constraint: it is cross-sectional. There is no follow-up, no randomized intervention, and no future outcome label that would justify a prognostic framing. Under these conditions, the most defensible scientific aim is not prediction of future decline, but clinically meaningful phenotyping anchored in an interpretable IC framework and evaluated against related functional correlates.

We therefore developed a reusable pipeline for cross-sectional IC phenotyping in older adults. Our approach had four goals: first, to replace binary IC impairment counts with continuous-domain IC approximation; second, to compare a strong linear baseline with clinically grouped masked tabular representation learning and semantic enhancement ablations; third, to identify reproducible unsupervised phenotypes from multidimensional assessment data; and fourth, to test whether those phenotypes showed coherent gradients across held-out functional correlates such as ADL, IADL, sarcopenia, frailty, fall risk, resilience, social function, and quality-of-life indicators.

## Research Design and Methods

### Data Source and Participants

We analyzed a multidimensional geriatric assessment export containing 6,025 unique participants and 361 raw variables. Participant ages ranged from 60 to 100 years. The dataset contained one row per participant, and no fully duplicated records were identified. Birth date and age fields suggested that the assessments were concentrated in mid-2024, consistent with a single cross-sectional baseline-like collection period rather than follow-up.

Because the study question was phenotyping rather than prediction, no longitudinal outcomes were defined. The analysis focused on cross-sectional heterogeneity in IC-related function and its association with related functional correlates.

### Data Cleaning and Feature Selection

Explicit missing-value encodings, including ND, UK, NA, and empty strings, were normalized to missing. Missingness was highly uneven across variables: 206 columns had at most 5% missingness, whereas 51 columns had 95%-100% missingness, largely representing rare disease duration or treatment-status items. These ultra-sparse variables were excluded from the primary modeling matrix because they added noise without reliable sample-wide information.

For the structured phenotyping pipeline, numeric features were median-imputed after clinically simple winsorization of selected count variables at the 1st and 99th percentiles. Categorical variables were mapped using the project codebook, coded with an explicit missing category when needed, and low-frequency categories were collapsed into "Other." Numeric variables with at least 5% missingness received missingness indicators. Province, city, and district fields were retained for exploratory descriptive work but not entered into the primary model because their missingness approached 28%-30% and could have introduced unstable center-related structure.

External functional validators were deliberately excluded from the clustering design matrix. These variables were reserved for downstream clinical interpretation and should be considered related functional correlates rather than fully independent end points because some constructs partially overlap with the domains used to approximate IC.

### Continuous-Domain Approximation of Intrinsic Capacity

We constructed five continuous IC domains, each scaled from 0 to 100, using clinically informed approximations based on available assessment items.

Cognition was approximated from the cognitive total score on a 0-30 scale. Psychological capacity was derived from the psychological total score, with lower symptom burden mapping to higher capacity, and an additional penalty applied for recorded anxiety/depression diagnosis. Vitality combined vitality total score, nutritional description, BMI deviation from a reference value, and calf circumference. Locomotion combined locomotion total score, test completion, gait abnormality, 4-meter gait speed, and grip strength, with sex-specific grip reference thresholds. Sensory capacity integrated hearing and vision impairment status, screening results, and whether impairment affected daily life. Overall continuous IC was defined as the mean of the five domain scores.

This construction should be interpreted as a clinically informed approximation of IC, not as a validated gold-standard IC instrument. The goal was to preserve within-domain severity and multidimensional balance better than binary impairment counts.

### Representation Learning and Clustering

The primary structured feature set contained 59 core variables from demographics, self-rated health, cognition, psychological status, vitality, locomotion, sensory status, disease burden, and lifestyle, plus the five continuous IC domain scores. After preprocessing, the design matrix contained 175 features grouped into nine clinically meaningful blocks.

We compared three representation strategies. The first was PCA to 128 dimensions, used as a strong linear baseline. The second was GroupMaskedFT, a group-aware masked tabular encoder that treated clinical blocks as grouped tokens, randomly masked approximately 30% of groups during training, and reconstructed only the masked groups to learn cross-domain structure. The third comprised dual-tower ablations in which a medicalized text representation of each participant was encoded by a Chinese BERT model and aligned with the tabular encoder through contrastive learning. These semantic models were evaluated as enhancement strategies rather than presumed primary solutions.

For each representation, we evaluated KMeans and Gaussian mixture clustering with k values from 3 to 6. Candidate solutions were scored using silhouette, Calinski-Harabasz index, Davies-Bouldin index, bootstrap stability, minimum cluster ratio, and separation across held-out functional correlates. For the manuscript, we prioritized solutions that combined reproducibility, balanced cluster sizes, clinically coherent gradients, and avoidance of clusters defined predominantly by a single demographic variable.

### External Clinical Correlates and Marker Discovery

Held-out functional correlates included ADL total, IADL total, sarcopenia score, Fried frailty score, frailty screen score, fall risk score, resilience score, social function score, and selected quality-of-life items. Group differences were evaluated using Kruskal-Wallis tests, and effect size was summarized as eta squared. For interpretability, we also ranked candidate phenotype-differentiating markers among clinically legible variables such as cognition total, self-rated health, locomotion total, psychological total, vitality total, gait abnormality, age, gait speed, and grip strength.

### Sensitivity Analyses

Three sensitivity analyses were prespecified. First, we compared the primary deep solution with the PCA baseline to assess how much added value came from the deep tabular encoder versus the underlying low-rank structure of the data. Second, we evaluated dual-tower semantic enhancement models to determine whether text-table alignment improved phenotyping. Third, we examined a higher-granularity 4-cluster Gaussian mixture solution from the GroupMaskedFT embedding. Because this solution yielded a sex-dominant split within the high-capacity group, it was treated as supplementary rather than primary.

## Results

### Sample Characteristics and Continuous IC Structure

The analytic sample included 6,025 older adults with a mean age of 71.1 years (SD 6.9); 2,816 (46.7%) were men and 3,209 (53.3%) were women. The mean self-rated health score was 6.68, the mean number of chronic conditions was 1.50, and the mean number of medications was 2.07. Missingness was manageable for most clinically central variables, but a substantial tail of rare disease duration and treatment-status variables was nearly empty and therefore excluded from the primary representation matrix.

Mean overall continuous IC was 74.5/100. The lowest mean domain scores were locomotion (62.3) and sensory function (63.4), whereas cognition (81.6), psychological capacity (81.4), and vitality (83.8) were comparatively higher. This pattern suggested that mobility and sensory function were the dominant areas of vulnerability in the overall cohort before any clustering was performed.

### Comparison of Candidate Representation Strategies

Among the best solution identified for each candidate representation strategy, GroupMaskedFT with KMeans at k = 3 showed the clearest latent geometry, with a silhouette of 0.311, stability ARI of 0.988, minimum cluster ratio of 0.240, and external separation of 0.217. The PCA baseline with KMeans at k = 3 had a much lower silhouette of 0.110 but similar stability (0.991), acceptable balance (minimum cluster ratio 0.223), and slightly higher external separation (0.225). Dual-tower semantic enhancement models did not outperform the primary tabular encoder; their external separation ranged from 0.138 to 0.162.

These results led to a deliberately conservative interpretation. The deep tabular encoder clearly improved latent organization, but the PCA baseline remained clinically competitive. We therefore used GroupMaskedFT + KMeans (k = 3) as the primary manuscript solution because it provided substantially clearer structure while preserving reproducibility and balanced cluster sizes, and we retained PCA as the principal comparator rather than treating it as a weak formality.

### Three Functional Phenotypes in the Primary GroupMaskedFT Solution

The primary solution yielded three phenotypes of sizes 1,571 (26.1%), 1,445 (24.0%), and 3,009 (49.9%). Sex balance was preserved across the primary clusters rather than dominating them: the proportions of men were 43.2%, 41.4%, and 51.2%, respectively.

The first phenotype represented a global low-capacity pattern. It was the oldest group (mean age 75.2 years), had the lowest self-rated health (mean 5.0), and showed the lowest overall IC (61.4). The deepest deficits were observed in locomotion (40.5), cognition (64.4), and psychological capacity (67.7), while vitality (76.1) remained relatively less impaired than mobility.

The second phenotype was intermediate in overall IC (73.1) and age (71.6 years) but retained marked vulnerability in locomotion (60.8) and sensory function (59.7). Cognition (80.8), psychological capacity (81.0), and vitality (83.4) were clearly better than in the low-capacity phenotype, suggesting a transitional or partially compensated pattern rather than global dysfunction.

The third phenotype represented the highest-capacity group. It was the youngest cluster (mean age 68.7 years), had the best self-rated health (7.66), lowest chronic disease burden (1.30), and lowest medication count (1.77). Its overall IC reached 82.0, with relatively preserved locomotion (74.3) and sensory function (67.9) compared with the other two groups.

Taken together, the three-cluster structure resembled an ordered functional spectrum with subtype-like differences in the relative persistence of locomotion and sensory vulnerability, rather than three wholly distinct mechanistic disease classes.

### External Clinical Correlates of the Three Phenotypes

The phenotypes showed strong and highly consistent gradients across held-out functional correlates. ADL total increased from 83.3 in the low-capacity phenotype to 95.7 in the intermediate phenotype and 99.2 in the high-capacity phenotype. IADL dependence score declined from 13.6 to 9.7 to 8.2 across the same sequence. Sarcopenia score decreased from 3.28 to 1.22 to 0.38, Fried frailty score from 2.00 to 1.24 to 0.72, frailty screen score from 1.47 to 0.53 to 0.21, and fall risk score from 36.8 to 25.3 to 17.2. Resilience score increased steadily from 39.6 to 44.8 to 50.3.

The largest effect sizes were observed for sarcopenia score (eta squared 0.337), IADL total (0.320), frailty screen score (0.284), Fried frailty score (0.259), QoL general health item (0.254), and ADL total (0.228). Social function, resilience, energy, social limitation, fall risk, and depressed mood also differed significantly across the three phenotypes.

These associations support the clinical relevance of the phenotype solution. The external variables were not fully independent of the underlying functional domains, but the consistent ordering across multiple related constructs suggests that the clusters were not arbitrary mathematical partitions.

### Phenotype-Differentiating Markers

The strongest phenotype-differentiating markers were cognition total (eta squared 0.415), self-rated health (0.325), locomotion total (0.300), psychological total (0.277), gait abnormality (0.184), vitality total (0.180), and age (0.153). Grip strength and gait speed also contributed, although with smaller effect sizes.

This ranking reinforced the view that the major axis of heterogeneity was not a single disease label but a composite of cognitive status, mobility, subjective health perception, and psychological burden, with locomotion playing the most visibly capacity-limiting role.

### Sensitivity Analyses

The PCA baseline reproduced a similar low-intermediate-high ordering. Its three phenotypes showed overall IC means of 59.7, 77.9, and 79.1 and similar gradients in ADL, IADL, sarcopenia, frailty, and fall risk. This supported the robustness of the overall phenotyping signal and confirmed that the dataset itself contains strong low-rank structure.

Dual-tower semantic enhancement did not improve phenotyping quality over the primary group-aware masked tabular encoder. In this dataset, text generation appeared to reorganize already structured information rather than add stable discriminative content.

A 4-cluster GroupMaskedFT Gaussian mixture solution slightly increased external separation to 0.238 and appeared to split the high-capacity range more finely. However, two of its high-capacity clusters were almost completely separated by sex, with one cluster being 99.8% male and another 98.3% female. We therefore treated this solution as a higher-granularity sensitivity analysis rather than a primary phenotype structure.

## Discussion

In this cross-sectional geriatric assessment dataset, continuous-domain IC approximation combined with group-aware masked representation learning recovered a coherent and clinically meaningful spectrum of functional heterogeneity. Three main findings deserve emphasis.

First, replacing binary IC impairment counts with continuous-domain IC scores substantially improved the clinical interpretability of the data structure. The cohort-level pattern already showed that locomotion and sensory function were the most vulnerable domains, and the cluster solution built on this gradient rather than collapsing participants into a simple tally of impaired domains. This is important because many older adults with similar global IC counts may differ substantially in how deficits are distributed across domains.

Second, the primary phenotype structure was clinically legible. The low-capacity phenotype concentrated the oldest participants and the worst ADL, IADL, sarcopenia, frailty, and fall-risk profiles. The intermediate phenotype was not merely "average"; it retained persistent locomotion-sensory vulnerability despite much better global function than the low-capacity group. The high-capacity phenotype showed the most preserved multidomain profile and the strongest resilience-related pattern. This ordered structure suggests that in cross-sectional assessment data, functional heterogeneity may often be best understood as a gradient with domain-specific accents, rather than a set of sharply discrete syndromic entities.

Third, the deep tabular model added value, but not in a simplistic "deep beats linear" sense. GroupMaskedFT produced far clearer latent geometry than PCA and offered a principled way to learn cross-domain dependencies through group masking. At the same time, PCA remained a very strong comparator and even showed slightly stronger separation on held-out functional correlates in its own best solution. We regard this not as a weakness, but as an important safeguard against overclaiming. In geriatric phenotyping with structured assessment data, stronger latent organization does not automatically translate into universal clinical superiority, and honest benchmarking against linear baselines should be preserved.

The marker ranking provides a useful interpretive frame. Cognition, self-rated health, locomotion total, psychological burden, gait abnormality, vitality, and age were the most important differentiators, suggesting that IC heterogeneity in this cohort was organized around a combined mobility-cognition-mental health axis rather than any single disease category. The prominence of locomotion is especially noteworthy. It was both the lowest-scoring cohort-level domain and a major separator across phenotypes, consistent with the centrality of mobility in broader geriatric vulnerability.

The semantic enhancement branch also yielded a useful negative result. Text-table dual-tower learning did not outperform the primary tabular encoder, suggesting that the current Tab2Text representation largely restated information already available in the structured features. This does not eliminate the future value of multimodal representation learning, but it argues against using text augmentation as a headline result in the present dataset.

This study has important limitations. The data were cross-sectional, so we cannot infer true trajectories, stage transitions, or future risk. Our IC scores were clinically informed approximations rather than validated gold-standard IC measurements. The external correlates were deliberately held out from clustering, but several remain conceptually related to the same functional domains and should not be interpreted as wholly independent outcomes. We had no external cohort for replication, and geographic fields had substantial missingness, limiting any strong claims about regional heterogeneity. Finally, the phenotype structure should be interpreted as functional phenotyping, not causal subtype discovery.

Despite these limitations, the study also has practical strengths. It demonstrates a reusable workflow for turning high-dimensional, mixed-format geriatric assessment data into interpretable continuous IC phenotypes without relying on future labels. It also shows how to keep the analysis clinically honest: preserve strong baselines, separate primary and supplementary solutions, and resist the temptation to present every higher-granularity split as biologically meaningful.

## Conclusion

A continuous-domain approximation of intrinsic capacity combined with group-aware masked tabular representation learning identified three clinically meaningful functional phenotypes in 6,025 older adults from a cross-sectional multidimensional assessment dataset. The primary phenotype solution showed strong gradients across ADL, IADL, sarcopenia, frailty, fall risk, resilience, and quality-of-life indicators. Deep tabular representation improved latent structure, but a strong PCA baseline remained clinically competitive, underscoring the need for balanced interpretation. In cross-sectional geriatric research, the most defensible contribution of these methods is not prognosis, but careful phenotyping of multidomain functional heterogeneity that can guide future longitudinal validation and targeted intervention studies.
