const fs = require('fs');

// Languages to update
const languages = ['de', 'zh', 'ja', 'pt', 'ar'];

// Sections to add
const sections = [
  'addFamilyMember', 'analytics', 'audit', 'createLesionGroup',
  'dermatologist', 'dermoscopy', 'familyHistory', 'geneticRisk',
  'geneticTesting', 'help', 'lesionDetail', 'lesionTracking',
  'regulatory', 'treatmentMonitoring'
];

// Read all section data from English
const sectionData = {};
sections.forEach(section => {
  sectionData[section] = JSON.parse(fs.readFileSync(`section_${section}.json`, 'utf8'));
});

// Process each language file
languages.forEach(lang => {
  const filename = `${lang}.json`;
  console.log(`\nProcessing ${filename}...`);

  // Read the existing file
  const data = JSON.parse(fs.readFileSync(filename, 'utf8'));

  // Add each missing section
  let addedCount = 0;
  sections.forEach(section => {
    if (!data[section]) {
      data[section] = sectionData[section];
      addedCount++;
      console.log(`  Added: ${section}`);
    } else {
      console.log(`  Skipped (exists): ${section}`);
    }
  });

  // Write the updated file with proper formatting
  fs.writeFileSync(filename, JSON.stringify(data, null, 2), 'utf8');
  console.log(`✓ ${lang}.json updated - Added ${addedCount} sections`);
});

console.log('\n=== All language files updated successfully! ===');
