class Settings {
  constructor() {
    this.settings = {
      model: 'gpt-3.5-turbo',
      temperature: 0.7,
      maxTokens: 2000
    };
    this.init();
  }

  async init() {
    try {
      const response = await fetch('/api/settings');
      const data = await response.json();
      this.settings = data;
      this.updateUI();
    } catch (error) {
      console.error('Failed to load settings:', error);
    }
  }

  async updateSettings(newSettings) {
    try {
      const response = await fetch('/api/settings', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(newSettings)
      });
      
      const data = await response.json();
      if (data.status === 'success') {
        this.settings = { ...this.settings, ...newSettings };
        this.updateUI();
      }
    } catch (error) {
      console.error('Failed to update settings:', error);
    }
  }

  updateUI() {
    // 更新設置界面的UI元素
    const modelSelect = document.getElementById('model-select');
    const temperatureSlider = document.getElementById('temperature-slider');
    const maxTokensInput = document.getElementById('max-tokens-input');

    if (modelSelect) modelSelect.value = this.settings.model;
    if (temperatureSlider) temperatureSlider.value = this.settings.temperature;
    if (maxTokensInput) maxTokensInput.value = this.settings.maxTokens;
  }

  getSettings() {
    return this.settings;
  }
}

// 導出設置實例
export const settings = new Settings();