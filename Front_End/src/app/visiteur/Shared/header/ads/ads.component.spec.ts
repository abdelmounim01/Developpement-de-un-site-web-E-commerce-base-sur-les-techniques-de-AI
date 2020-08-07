import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { AdsvisiteurComponent } from './ads.component';

describe('AdsComponent', () => {
  let component: AdsvisiteurComponent;
  let fixture: ComponentFixture<AdsvisiteurComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ AdsvisiteurComponent ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(AdsvisiteurComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
