import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { AdsHomevisiteurComponent } from './ads-home.component';

describe('AdsHomeComponent', () => {
  let component: AdsHomevisiteurComponent;
  let fixture: ComponentFixture<AdsHomevisiteurComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ AdsHomevisiteurComponent ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(AdsHomevisiteurComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
