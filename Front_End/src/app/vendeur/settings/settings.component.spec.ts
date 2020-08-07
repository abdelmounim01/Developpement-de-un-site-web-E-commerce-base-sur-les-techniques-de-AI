import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { SettingsComponentAdmin } from './settings.component';

describe('SettingsComponent', () => {
  let component: SettingsComponentAdmin;
  let fixture: ComponentFixture<SettingsComponentAdmin>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ SettingsComponentAdmin ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(SettingsComponentAdmin);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
